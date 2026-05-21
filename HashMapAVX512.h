#include <immintrin.h>
#include <bit>

#include "HashMapImpl.h"

//---------------------------------------------------------------------------------
template< typename V >
class alignas( 64 ) HashMap64AVX512 : public HashMap64ImplBase< V >
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
uint32_t HashMap64AVX512< V >::GetCount( ) const
{
  uint32_t count = 0;
  for ( uint32_t i_grp = 0; i_grp < this->m_GroupCount; ++i_grp )
  {
    __m512i  used_bit    = _mm512_set1_epi32 ( HashMap64ImplBase< V >::kIsNotEmpty );
    __m512i  meta_data   = _mm512_loadu_epi32( &this->m_Meta[ i_grp ] );
    __m512i  used_meta   = _mm512_and_epi32  ( meta_data, used_bit );
    uint16_t used_bitset = _mm512_cmp_epi32_mask( used_meta, used_bit, _MM_CMPINT_EQ );

    count += __popcnt16( used_bitset );
  }

  return count;
}

//---------------------------------------------------------------------------------
template< typename V >
void HashMap64AVX512< V >::Insert( uint64_t key, const V& value )
{
  if ( V* val = Insert( key ) )
  {
    *val = value;
  }
}

//---------------------------------------------------------------------------------
template< typename V >
V* HashMap64AVX512< V >::Insert( uint64_t key )
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
  __m512i  not_empty_mask    = _mm512_set1_epi32( HashMap64ImplBase< V >::kIsNotEmpty );
  __m512i  potential_dup_val = _mm512_set1_epi32( ctrl_to_insert );

  uint16_t unavailable_slots_bitset = 0xffff;
  uint16_t empty_slot               = 0xffff;

  do
  {
    __m512i  meta_data                = _mm512_loadu_epi32   ( meta );
    __m512i  not_empty                = _mm512_and_epi32     ( meta_data, not_empty_mask );
    uint16_t used_bitset              = _mm512_cmp_epi32_mask( not_empty, _mm512_setzero_si512(), _MM_CMPINT_NE );
             unavailable_slots_bitset = ( used_bitset & ctrl_mask ) | ~ctrl_mask;
             empty_slot               = std::countr_one( unavailable_slots_bitset );

    // Check for duplicates
    // truncate potential duplicates to only within available range
    uint16_t dup_check_bitset = ~(uint16_t)(0xffff << empty_slot);

    uint16_t potential_dup_bitset = _mm512_cmp_epi32_mask( meta_data, potential_dup_val, _MM_CMPINT_EQ );
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
V* HashMap64AVX512< V >::At( uint64_t key )
{
  const V* v = const_cast< const HashMap64AVX512<V>* >(this)->At( key );
  return const_cast<V*>( v );
}

//---------------------------------------------------------------------------------
template< typename V >
const V* HashMap64AVX512< V >::At( uint64_t key ) const
{
  ControlT ctrl_cmp         = HashMap64ImplBase< V >::kIsNotEmpty | HashMap64ImplBase< V >::CtrlFor( key );
  uint32_t start_idx        = HashMap64ImplBase< V >::HashFunction( key ) % this->m_Capacity;
  uint32_t group_idx_cyclic = start_idx / HashMap64ImplBase< V >::kElemsPerMeta;
  uint32_t ctrl_idx         = start_idx - ( group_idx_cyclic * HashMap64ImplBase< V >::kElemsPerMeta );
  
  while ( group_idx_cyclic < this->m_GroupCount )
  {
    uint32_t group_idx     = group_idx_cyclic % this->m_GroupCount;
    Group*   meta          = &this->m_Meta[ group_idx % this->m_GroupCount ];
    // If anything has been set here before, it's a tombstone sentinel whether or not the empty bit is set
    __m512i  meta_data     = _mm512_loadu_epi32( meta );
    uint16_t used_bitset   = _mm512_cmp_epi32_mask( meta_data, _mm512_setzero_si512(), _MM_CMPINT_NE );

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
void HashMap64AVX512< V >::Remove( uint64_t key )
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
    __m512i  meta_data   = _mm512_loadu_epi32( meta );
    uint16_t used_bitset = _mm512_cmp_epi32_mask( meta_data, _mm512_setzero_si512(), _MM_CMPINT_NE );

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
bool HashMap64AVX512< V >::Contains( uint64_t key ) const
{
  return At( key ) != nullptr;
}

//---------------------------------------------------------------------------------
template< typename V >
bool HashMap64AVX512< V >::IsFull() const
{
  uint32_t count = 0;
  for ( uint32_t i_grp = 0; i_grp < this->m_GroupCount; ++i_grp )
  {
    __m512i  used_bit    = _mm512_set1_epi32 ( HashMap64ImplBase< V >::kIsNotEmpty );
    __m512i  meta_data   = _mm512_loadu_epi32( &this->m_Meta[ i_grp ] );
    __m512i  used_meta   = _mm512_and_epi32  ( meta_data, used_bit );
    uint16_t used_bitset = _mm512_cmp_epi32_mask( used_meta, used_bit, _MM_CMPINT_EQ );

    if ( used_bitset != 0xffff )
    {
      return false;
    }
  }

  return true;
}

//---------------------------------------------------------------------------------
template< typename V >
bool HashMap64AVX512< V >::IsEmpty() const
{
  for ( uint32_t i_grp = 0; i_grp < this->m_GroupCount; ++i_grp )
  {
    __m512i  used_bit    = _mm512_set1_epi32 ( HashMap64ImplBase< V >::kIsNotEmpty );
    __m512i  meta_data   = _mm512_loadu_epi32( &this->m_Meta[ i_grp ] );
    __m512i  used_meta   = _mm512_and_epi32  ( meta_data, used_bit );
    uint16_t used_bitset = _mm512_cmp_epi32_mask( used_meta, used_bit, _MM_CMPINT_EQ );

    if ( used_bitset > 0 )
    {
      return false;
    }
  }

  return true;
}

//---------------------------------------------------------------------------------
template< typename V >
uint64_t HashMap64AVX512< V >::GetFirstKey()
{
  for ( uint32_t i_grp = 0; i_grp < this->m_GroupCount; ++i_grp )
  {
    __m512i  used_bit      = _mm512_set1_epi32 ( HashMap64ImplBase< V >::kIsNotEmpty );
    __m512i  meta_data     = _mm512_loadu_epi32( &this->m_Meta[ i_grp ] );
    __m512i  used_meta     = _mm512_and_epi32  ( meta_data, used_bit );
    uint16_t used_bitset   = _mm512_cmp_epi32_mask( used_meta, used_bit, _MM_CMPINT_EQ );

    if ( used_bitset > 0 )
    {
      uint32_t first_idx = std::countr_zero( used_bitset );
      return this->m_Elems[ i_grp * HashMap64ImplBase< V >::kElemsPerMeta + first_idx ].m_Key;
    }
  }

  return 0;
}