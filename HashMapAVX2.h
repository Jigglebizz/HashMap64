#include <immintrin.h>
#include <bit>

#include "HashMapImpl.h"

//---------------------------------------------------------------------------------
template< typename V >
class alignas( 64 ) HashMap64AVX2 : public HashMap64ImplBase< V >
{
public:
  void                    Insert          ( uint64_t key, const V& value ) override;
  V*                      Insert          ( uint64_t key ) override;
  V*                      At              ( uint64_t key ) override;
  const V*                At              ( uint64_t key ) const override;
  void                    Remove          ( uint64_t key ) override;
  uint32_t                GetCount        ( ) const override;
                          
  bool                    IsFull          ( ) const override;
  bool                    IsEmpty         ( ) const override;
  bool                    Contains        ( uint64_t key ) const override;
  uint64_t                GetFirstKey     ( ) override;

private:
  using ControlT = typename HashMap64ImplBase< V >::ControlT;
  using Group    = typename HashMap64ImplBase< V >::Group;
  using Elem     = typename HashMap64ImplBase< V >::Elem;
};

//---------------------------------------------------------------------------------
template< typename V >
uint32_t HashMap64AVX2< V >::GetCount( ) const
{
  uint32_t count = 0;
  for ( uint32_t i_grp = 0; i_grp < this->m_GroupCount; ++i_grp )
  {
    static_assert( HashMap64ImplBase< V >::kIsNotEmpty == 0x80000000, "Assuming these are equal for this algorithm" );

    __m256i  meta_data1   = _mm256_loadu_si256( (const __m256i*)&this->m_Meta[ i_grp ] );
    __m256i  meta_data2   = _mm256_loadu_si256( (const __m256i*)&this->m_Meta[ i_grp ].m_Control[ HashMap64ImplBase< V >::kElemsPerMeta / 2 ]);
    uint32_t used_bitset1 = _mm256_movemask_epi8( meta_data1 );
    uint32_t used_bitset2 = _mm256_movemask_epi8( meta_data2 );
             used_bitset1 = _pext_u32( used_bitset1, 0x88888888 );
             used_bitset2 = _pext_u32( used_bitset2, 0x88888888 );

    count += __popcnt( used_bitset1 ) + __popcnt( used_bitset2 );
  }

  return count;
}

//---------------------------------------------------------------------------------
template< typename V >
void HashMap64AVX2< V >::Insert( uint64_t key, const V& value )
{
  if ( V* val = Insert( key ) )
  {
    *val = value;
  }
}

//---------------------------------------------------------------------------------
template< typename V >
V* HashMap64AVX2< V >::Insert( uint64_t key )
{
  ASSERT_MSG( key != 0, "Can not insert a key of 0" );

  ControlT        ctrl_to_insert  = HashMap64ImplBase< V >::CtrlFor( key ) | HashMap64ImplBase< V >::kIsNotEmpty;
  uint32_t        desired_idx     = HashMap64ImplBase< V >::HashFunction( key ) % this->m_Capacity;
  const uint32_t  start_group_idx = desired_idx / HashMap64ImplBase<V>::kElemsPerMeta;
  uint32_t        group_idx       = start_group_idx;
  uint32_t        ctrl_idx_offset = 0; // from desired_idx
  uint32_t        idx             = desired_idx;

  ctrl_idx_offset -= desired_idx % HashMap64ImplBase<V>::kElemsPerMeta;

  // find empty slot
  Group*   meta              = &this->m_Meta[ group_idx ];
  uint32_t ctrl_idx          = idx % HashMap64ImplBase< V >::kElemsPerMeta;
  uint16_t ctrl_mask         = (uint16_t)(0xffff << ctrl_idx);
  __m256i  not_empty_mask    = _mm256_set1_epi32( HashMap64ImplBase< V >::kIsNotEmpty );
  __m256i  potential_dup_val = _mm256_set1_epi32( ctrl_to_insert );
  __m256i  zero_lane         = _mm256_setzero_si256();

  uint16_t unavailable_slots_bitset = 0xffff;
  uint16_t empty_slot               = 0xffff;

  do
  {
    __m256i  meta_data1    = _mm256_loadu_epi32   ( meta );
    __m256i  not_empty1    = _mm256_and_epi32     ( meta_data1, not_empty_mask );
    __m256i  ne_eq_0_1     = _mm256_cmpeq_epi32   ( not_empty1, zero_lane );
    uint32_t used_bitset1  = _mm256_movemask_epi8 ( ne_eq_0_1 );
    __m256i  meta_data2    = _mm256_loadu_epi32   ( &meta->m_Control[ HashMap64ImplBase< V >::kElemsPerMeta / 2 ] );
    __m256i  not_empty2    = _mm256_and_epi32     ( meta_data2, not_empty_mask );
    __m256i  ne_eq_0_2     = _mm256_cmpeq_epi32   ( not_empty2, zero_lane );
    uint32_t used_bitset2  = _mm256_movemask_epi8 ( ne_eq_0_2 );
    uint64_t used_bitset3  = (uint64_t)used_bitset1 | ( (uint64_t)used_bitset2 << 32 );
    uint16_t used_bitset   = ~(uint16_t)_pext_u64( used_bitset3, 0x8888888888888888 );

             unavailable_slots_bitset = ( used_bitset & ctrl_mask ) | ~ctrl_mask;
             empty_slot               = std::countr_one( unavailable_slots_bitset );

    // Check for duplicates
    // truncate potential duplicates to only within available range
    uint16_t dup_check_bitset = ~(uint16_t)(0xffff << empty_slot);

    __m256i  potential_dup_1 = _mm256_cmpeq_epi32   ( meta_data1, potential_dup_val );
    __m256i  potential_dup_2 = _mm256_cmpeq_epi32   ( meta_data2, potential_dup_val );
    uint32_t dup_bitset_1    = _mm256_movemask_epi8 ( potential_dup_1 );
    uint32_t dup_bitset_2    = _mm256_movemask_epi8 ( potential_dup_2 );
    uint16_t potential_dup_bitset = (uint16_t)( _pext_u32( dup_bitset_1, 0x88888888 ) | ( _pext_u32( dup_bitset_2, 0x88888888 ) << 8) );

    potential_dup_bitset &= dup_check_bitset;

    while ( potential_dup_bitset )
    {
      uint16_t potential_dup_idx = std::countr_zero( potential_dup_bitset );
      uint32_t dup_elem_idx = ( group_idx % this->m_GroupCount ) * HashMap64ImplBase<V>::kElemsPerMeta + potential_dup_idx;
      if ( this->m_Elems[ dup_elem_idx ].m_Key == key )
      {
        return nullptr;
      }
      potential_dup_bitset &= ~( 1 << potential_dup_idx );
    }

    // All slots full, let's move along
    if ( unavailable_slots_bitset == 0xffff )
    {
      group_idx++;
      ctrl_idx_offset += HashMap64ImplBase< V >::kElemsPerMeta;
    }

    ctrl_mask = 0xffff;
    ctrl_idx  = 0;
    meta      = &this->m_Meta[ group_idx % this->m_GroupCount ];
  } while ( unavailable_slots_bitset == 0xffff && ctrl_idx_offset < this->m_Capacity );

  // breaking out of the loop means we've either found an empty space, or exceeded capacity
  ctrl_idx_offset += empty_slot;
  if ( ctrl_idx_offset >= this->m_Capacity )
  {
    return nullptr;
  }
  idx = ( ( desired_idx ) + ctrl_idx_offset ) % this->m_Capacity;

  // do insertion
  this->m_Meta[ group_idx % this->m_GroupCount ].m_Control[ idx % HashMap64ImplBase< V >::kElemsPerMeta ] = ctrl_to_insert;
  this->m_Elems[ idx ].m_Key = key;

  return &this->m_Elems[ idx ].m_Value;
}

//---------------------------------------------------------------------------------
template< typename V >
V* HashMap64AVX2< V >::At( uint64_t key )
{
  const V* v = const_cast< const HashMap64AVX2<V>* >(this)->At( key );
  return const_cast<V*>( v );
}

//---------------------------------------------------------------------------------
template< typename V >
const V* HashMap64AVX2< V >::At( uint64_t key ) const
{
  ControlT ctrl_cmp         = HashMap64ImplBase< V >::kIsNotEmpty | HashMap64ImplBase< V >::CtrlFor( key );
  uint32_t start_idx        = HashMap64ImplBase< V >::HashFunction( key ) % this->m_Capacity;
  uint32_t group_idx_cyclic = start_idx / HashMap64ImplBase< V >::kElemsPerMeta;
  uint32_t ctrl_idx         = start_idx - ( group_idx_cyclic * HashMap64ImplBase< V >::kElemsPerMeta );
  
  __m256i  zero_lane    = _mm256_setzero_si256();
  while ( group_idx_cyclic < this->m_GroupCount )
  {
    uint32_t group_idx     = group_idx_cyclic % this->m_GroupCount;
    Group*   meta          = &this->m_Meta[ group_idx % this->m_GroupCount ];
    // If anything has been set here before, it's a tombstone sentinel whether or not the empty bit is set
    __m256i  meta_data1   = _mm256_loadu_si256( (const __m256i*)meta );
    __m256i  meta_data2   = _mm256_loadu_si256( (const __m256i*)&meta->m_Control[ HashMap64ImplBase< V >::kElemsPerMeta / 2 ] );
    __m256i  meta_eq_0_1  = _mm256_cmpeq_epi32( meta_data1, zero_lane );
    __m256i  meta_eq_0_2  = _mm256_cmpeq_epi32( meta_data2, zero_lane );
    uint32_t used_bitset1 = _mm256_movemask_epi8( meta_eq_0_1 );
    uint32_t used_bitset2 = _mm256_movemask_epi8( meta_eq_0_2 );
    uint16_t used_bitset  = ~(uint16_t)( _pext_u32( used_bitset1, 0x88888888 ) | ( _pext_u32( used_bitset2, 0x88888888 ) << 8) );

    for ( uint32_t i_ctrl = ctrl_idx; i_ctrl < 16; ++i_ctrl)
    {
      // This slot is not empty
      if ( _bextr_u32( used_bitset, i_ctrl, 1 ) )
      {
        // Last 7 bits of hash match, very potentially a match. Gonna do a cache invalidate here to check
        if ( meta->m_Control[ i_ctrl ] == ctrl_cmp )
        {
          Elem* elem = &this->m_Elems[ group_idx * HashMap64ImplBase< V >::kElemsPerMeta + i_ctrl ];
          if ( elem->m_Key == key )
          { 
            // We found it, return it!
            return &elem->m_Value;
          }
        }
      }
      else
      {
        return nullptr; // Found an empty slot, we're done
      }
    }

    group_idx_cyclic++;
    ctrl_idx = 0;
  }
  return nullptr;
}

//---------------------------------------------------------------------------------
template< typename V >
void HashMap64AVX2< V >::Remove( uint64_t key )
{
  ASSERT_MSG( key != 0, "Key of 0 is not valid!" );

  ControlT ctrl_cmp         = HashMap64ImplBase< V >::kIsNotEmpty | HashMap64ImplBase< V >::CtrlFor( key );
  uint32_t start_idx        = HashMap64ImplBase< V >::HashFunction( key ) % this->m_Capacity;
  uint32_t group_idx_cyclic = start_idx / HashMap64ImplBase< V >::kElemsPerMeta;
  uint32_t ctrl_idx         = start_idx - ( group_idx_cyclic * HashMap64ImplBase< V >::kElemsPerMeta );
  
  while ( group_idx_cyclic < this->m_GroupCount )
  {
    uint32_t group_idx   = group_idx_cyclic % this->m_GroupCount;
    Group*   meta        = &this->m_Meta[ group_idx ];
    // If anything has been set here before, it's a tombstone sentinel whether or not the empty bit is set
    __m256i  meta_data1   = _mm256_loadu_si256( (const __m256i*)meta );
    __m256i  meta_data2   = _mm256_loadu_si256( (const __m256i*)&meta->m_Control[ HashMap64ImplBase< V >::kElemsPerMeta / 2 ] );
    __m256i  zero_lane    = _mm256_setzero_si256();
    __m256i  meta_eq_0_1  = _mm256_cmpeq_epi32( meta_data1, zero_lane );
    __m256i  meta_eq_0_2  = _mm256_cmpeq_epi32( meta_data2, zero_lane );
    uint32_t used_bitset1 = _mm256_movemask_epi8( meta_eq_0_1 );
    uint32_t used_bitset2 = _mm256_movemask_epi8( meta_eq_0_2 );
    uint16_t used_bitset  = ~(uint16_t)( _pext_u32( used_bitset1, 0x88888888 ) | ( _pext_u32( used_bitset2, 0x88888888 ) << 8) );

    for ( uint32_t i_ctrl = ctrl_idx; i_ctrl < 16; ++i_ctrl)
    {
      // This slot is not empty
      if ( _bextr_u32( used_bitset, i_ctrl, 1 ) )
      {
        // Last 7 bits of hash match, very potentially a match. Gonna do a cache invalidate here to check
        if ( meta->m_Control[ i_ctrl ] == ctrl_cmp )
        {
          Elem* elem = &this->m_Elems[ group_idx * HashMap64ImplBase< V >::kElemsPerMeta + i_ctrl ];
          if ( elem->m_Key == key )
          { 
            // We found it, remove it!
            meta->m_Control[ i_ctrl ] &= ~HashMap64ImplBase< V >::kIsNotEmpty;
          }
        }
      }
      else
      {
        return; // Found an empty slot, we're done
      }
    }

    group_idx_cyclic++;
    ctrl_idx = 0;
  }
}

//---------------------------------------------------------------------------------
template< typename V >
bool HashMap64AVX2< V >::Contains( uint64_t key ) const
{
  return At( key ) != nullptr;
}

//---------------------------------------------------------------------------------
template< typename V >
bool HashMap64AVX2< V >::IsFull() const
{
  uint32_t count = 0;
  for ( uint32_t i_grp = 0; i_grp < this->m_GroupCount; ++i_grp )
  {
    static_assert( HashMap64ImplBase< V >::kIsNotEmpty == 0x80000000, "Assuming these are equal for this algorithm" );

    __m256i  meta_data1   = _mm256_loadu_si256( (const __m256i*)&this->m_Meta[ i_grp ] );
    __m256i  meta_data2   = _mm256_loadu_si256( (const __m256i*)&this->m_Meta[ i_grp ].m_Control[ HashMap64ImplBase< V >::kElemsPerMeta / 2 ]);
    uint32_t used_bitset1 = _mm256_movemask_epi8( meta_data1 );
    uint32_t used_bitset2 = _mm256_movemask_epi8( meta_data2 );
             used_bitset1 = _pext_u32( used_bitset1, 0x88888888 );
             used_bitset2 = _pext_u32( used_bitset2, 0x88888888 );

    if ( used_bitset1 != 0x8888 && used_bitset2 != 0x8888 )
    {
      return false;
    }
  }

  return true;
}

//---------------------------------------------------------------------------------
template< typename V >
bool HashMap64AVX2< V >::IsEmpty() const
{
  for ( uint32_t i_grp = 0; i_grp < this->m_GroupCount; ++i_grp )
  {
    static_assert( HashMap64ImplBase< V >::kIsNotEmpty == 0x80000000, "Assuming these are equal for this algorithm" );

    __m256i  meta_data1   = _mm256_loadu_si256( (const __m256i*)&this->m_Meta[ i_grp ] );
    __m256i  meta_data2   = _mm256_loadu_si256( (const __m256i*)&this->m_Meta[ i_grp ].m_Control[ HashMap64ImplBase< V >::kElemsPerMeta / 2 ]);
    uint32_t used_bitset1 = _mm256_movemask_epi8( meta_data1 );
    uint32_t used_bitset2 = _mm256_movemask_epi8( meta_data2 );
             used_bitset1 = _pext_u32( used_bitset1, 0x88888888 );
             used_bitset2 = _pext_u32( used_bitset2, 0x88888888 );

    if ( used_bitset1 > 0 || used_bitset2 > 0 )
    {
      return false;
    }
  }

  return true;
}

//---------------------------------------------------------------------------------
template< typename V >
uint64_t HashMap64AVX2< V >::GetFirstKey()
{
  for ( uint32_t i_grp = 0; i_grp < this->m_GroupCount; ++i_grp )
  {
    __m256i  used_bit     = _mm256_set1_epi32( HashMap64ImplBase< V >::kIsNotEmpty );
    __m256i  meta_data1   = _mm256_loadu_si256( (const __m256i*)&this->m_Meta[ i_grp ] );
    __m256i  meta_data2   = _mm256_loadu_si256( (const __m256i*)&this->m_Meta[ i_grp ].m_Control[ HashMap64ImplBase< V >::kElemsPerMeta / 2 ]);
    __m256i  used_meta1   = _mm256_and_epi32 ( meta_data1, used_bit );
    __m256i  used_meta2   = _mm256_and_epi32 ( meta_data2, used_bit );
    uint32_t used_bitset1 = _mm256_movemask_epi8( used_meta1 );
    uint32_t used_bitset2 = _mm256_movemask_epi8( used_meta2 );

    if ( used_bitset1 > 0 )
    {
      uint32_t first_idx = std::countr_zero( used_bitset1 ) / 4;
      return this->m_Elems[ i_grp * HashMap64ImplBase< V >::kElemsPerMeta + first_idx ].m_Key;
    }
    if ( used_bitset2 > 0 )
    {
      uint32_t first_idx = std::countr_zero( used_bitset2 ) / 4 + 8;
      return this->m_Elems[ i_grp * HashMap64ImplBase< V >::kElemsPerMeta + first_idx ].m_Key;
    }
  }

  return 0;
}