#pragma once

#include <stdint.h>
#include <immintrin.h>
#include <bit>

#ifndef ASSERT_MSG
#define ASSERT_MSG( cond, msg, ... ) if ( ( cond ) == false ) { __debugbreak(); }
#endif

#ifndef CORE_API
#define CORE_API
#endif

//---------------------------------------------------------------------------------
template< typename V >
class HashMap64
{
public:
  HashMap64() 
  : m_Backing   (nullptr )
  #ifdef USE_HEAP
  , m_OwningHeap( nullptr )
  #endif
  , m_Meta      ( nullptr )
  , m_Elems     ( nullptr )
  , m_Capacity  ( 0 ) 
  , m_GroupCount( 0 )
  {}

                   void     CORE_API InitWithBacking        ( void* backing, uint32_t capacity );
                   #ifdef USE_HEAP
                   void     CORE_API InitOnHeap             ( MemAllocHeap* heap, uint32_t desired_capacity );
                   #endif
                   void     CORE_API Destroy                ( );
                   
                   void     CORE_API Insert                 ( uint64_t key, const V& value );
                   V*       CORE_API Insert                 ( uint64_t key );
                   V*       CORE_API At                     ( uint64_t key );
                   const V* CORE_API At                     ( uint64_t key ) const;
                   void     CORE_API Remove                 ( uint64_t key );
                   uint32_t CORE_API GetCapacity            ( ) const;
                   uint32_t CORE_API GetCount               ( ) const;
  static constexpr size_t   CORE_API GetRequiredBackingSize ( uint32_t capacity );

                   bool     CORE_API IsFull                 ( ) const;
                   bool     CORE_API IsEmpty                ( ) const;
                   bool     CORE_API Contains               ( uint64_t key ) const;
                   uint64_t CORE_API GetFirstKey            ( );

                   bool     CORE_API IsInitialized          ( ) const;

private:
  static constexpr uint32_t kElemsPerMeta = 16;

  using ControlT = uint32_t;
  enum Control : ControlT
  {
    kIsNotEmpty = 0x8000'0000,
    kHashRem    = 0x7fff'ffff,
  };

  struct Group
  {
    ControlT m_Control [ kElemsPerMeta ];
  };
  static_assert( sizeof( Group ) == sizeof( Group::m_Control ), "Must be well-packed" );
  static_assert( sizeof( Group ) == 64, "Must be the size of a cacheline" );

  struct Elem
  {
    uint64_t m_Key;
    V        m_Value;
  };
  
  #ifdef USE_HEAP
  MemAllocHeap* m_OwningHeap;
  #endif
  void*         m_Backing;
  Group*        m_Meta;
  Elem*         m_Elems;
  uint32_t      m_Capacity;
  uint32_t      m_GroupCount;

  static inline ControlT CtrlFor( uint64_t key );
};

//---------------------------------------------------------------------------------
// Murmur64
static uint32_t HashFunction( uint64_t input )
{
  input ^= input >> 33;
  input *= 0xff51afd7ed558ccdULL;
  input ^= input >> 33;
  input *= 0xc4ceb9fe1a85ec53ULL;
  input ^= input >> 33;
  return (uint32_t)input;
}

//---------------------------------------------------------------------------------
template< typename V >
uint32_t HashMap64< V >::GetCapacity ( ) const
{
  return m_Capacity;
}

//---------------------------------------------------------------------------------
template< typename V >
uint32_t HashMap64< V >::GetCount( ) const
{
  uint32_t count = 0;
  for ( uint32_t i_grp = 0; i_grp < m_GroupCount; ++i_grp )
  {
    __m512i used_bit       = _mm512_set1_epi32 ( kIsNotEmpty );
    __m512i meta_data      = _mm512_loadu_epi32( &m_Meta[ i_grp ] );
    __m512i used_meta      = _mm512_and_epi32  ( meta_data, used_bit );
    uint16_t used_bitset   = _mm512_cmp_epi32_mask( used_meta, used_bit, _MM_CMPINT_EQ );

    count += __popcnt16( used_bitset );
  }

  return count;
}

//---------------------------------------------------------------------------------
template< typename V >
constexpr size_t HashMap64< V >::GetRequiredBackingSize( uint32_t capacity )
{
  return capacity * sizeof( Elem ) + ( capacity * sizeof( Group ) / kElemsPerMeta );
}

//---------------------------------------------------------------------------------
template< typename V >
void HashMap64< V >::InitWithBacking( void* backing, uint32_t capacity )
{
  ASSERT_MSG( capacity % kElemsPerMeta == 0 , "Capacity must be a multiple of 16" );

  m_Backing    = backing;
  m_Capacity   = capacity;
  m_GroupCount = capacity / kElemsPerMeta;

  size_t meta_size = ( sizeof( *m_Meta ) * m_GroupCount );
  m_Meta = (Group*)backing;

  memset( m_Meta, 0, meta_size );

  m_Elems = (Elem*)(((uint8_t*)backing) + meta_size);
}

#ifdef USE_HEAP
//---------------------------------------------------------------------------------
template< typename V >
void HashMap64< V >::InitOnHeap( MemAllocHeap* heap, uint32_t desired_capacity )
{
  size_t required_size = GetRequiredBackingSize( desired_capacity );
  m_Backing = (uint8_t*)heap->Alloc( required_size );
  ASSERT_MSG( m_Backing != nullptr, "Not enough space in heap for hashmap!" );

  InitWithBacking( m_Backing, desired_capacity, num_buckets );

  m_OwningHeap  = heap;
}
#endif

//---------------------------------------------------------------------------------
template< typename V >
void HashMap64< V >::Destroy()
{
#ifdef USE_HEAP
  if ( m_OwningHeap != nullptr )
  {
    m_OwningHeap->Free( m_Backing );
  }
#endif
}


//---------------------------------------------------------------------------------
template< typename V >
void HashMap64< V >::Insert( uint64_t key, const V& value )
{
  if ( V* val = Insert( key ) )
  {
    *val = value;
  }
}

//---------------------------------------------------------------------------------
template< typename V >
V* HashMap64< V >::Insert( uint64_t key )
{
  ASSERT_MSG( m_Backing != nullptr, "HashMap not initialized!" );
  ASSERT_MSG( key != 0,             "Can not insert a key of 0" );
  
  ControlT       ctrl_to_insert  = CtrlFor( key ) | kIsNotEmpty;
  uint32_t       desired_idx     = HashFunction( key ) % m_Capacity;
  const uint32_t start_group_idx = desired_idx / kElemsPerMeta;
  uint32_t       group_idx       = start_group_idx;
  uint32_t       ctrl_idx_offset = 0;
  uint32_t       idx             = ( desired_idx + ctrl_idx_offset ) % m_Capacity;

  // find empty slot
  ControlT ctrl = m_Meta[ group_idx % m_GroupCount ].m_Control[ idx % kElemsPerMeta ];
  while ( ctrl & kIsNotEmpty )
  {
    // Before moving on, we need to check if we're a duplicate of an existing key
    if ( ctrl == ctrl_to_insert )
    {
      if ( m_Elems[ idx ].m_Key == key )
      {
        return nullptr;
      }
    }

    ctrl_idx_offset++;
    if ( ctrl_idx_offset == m_Capacity )
    {
      return nullptr;
    }

    idx = ( desired_idx + ctrl_idx_offset ) % m_Capacity;
    group_idx += ( idx % kElemsPerMeta == 0 );
    ctrl = m_Meta[ group_idx % m_GroupCount ].m_Control[ idx % kElemsPerMeta ];
  }

  // do insertion
  m_Meta[ group_idx % m_GroupCount ].m_Control[ idx % kElemsPerMeta ] = ctrl_to_insert;
  m_Elems[ idx ].m_Key = key;

  return &m_Elems[ idx ].m_Value;
}

//---------------------------------------------------------------------------------
template< typename V >
V* HashMap64< V >::At( uint64_t key )
{
  const V* v = const_cast< const HashMap64<V>* >(this)->At( key );
  return const_cast<V*>( v );
}

//---------------------------------------------------------------------------------
template< typename V >
const V* HashMap64< V >::At( uint64_t key ) const
{
  ASSERT_MSG( m_Backing != nullptr, "HashMap not initialized!" );

  ControlT       ctrl_cmp         = kIsNotEmpty | CtrlFor( key );
  uint32_t       start_idx        = HashFunction( key ) % m_Capacity;
  uint32_t       group_idx_cyclic = start_idx / kElemsPerMeta;
  uint32_t       ctrl_idx         = start_idx - ( group_idx_cyclic * kElemsPerMeta );
  
  while ( group_idx_cyclic < m_GroupCount )
  {
    uint32_t group_idx = group_idx_cyclic % m_GroupCount;
    Group*   meta      = &m_Meta[ group_idx % m_GroupCount ];
    // If anything has been set here before, it's a tombstone sentinel whether or not the empty bit is set
    __m512i meta_data      = _mm512_loadu_epi32( meta );
    uint16_t used_bitset   = _mm512_cmp_epi32_mask( meta_data, _mm512_setzero_si512(), _MM_CMPINT_NE );

    for ( uint32_t i_ctrl = ctrl_idx; i_ctrl < 16; ++i_ctrl)
    {
      // This slot is not empty
      if ( _bextr_u32( used_bitset, i_ctrl, 1 ) )
      {
        // Last 7 bits of hash match, very potentially a match. Gonna do a cache invalidate here to check
        if ( meta->m_Control[ i_ctrl ] == ctrl_cmp )
        {
          Elem* elem = &m_Elems[ group_idx * kElemsPerMeta + i_ctrl ];
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
void HashMap64< V >::Remove( uint64_t key )
{
  ASSERT_MSG( m_Backing != nullptr, "HashMap not initialized!" );
  ASSERT_MSG( key != 0,             "Key of 0 is not valid!" );

  ControlT       ctrl_cmp         = kIsNotEmpty | CtrlFor( key );
  uint32_t       start_idx        = HashFunction( key ) % m_Capacity;
  uint32_t       group_idx_cyclic = start_idx / kElemsPerMeta;
  uint32_t       ctrl_idx         = start_idx - ( group_idx_cyclic * kElemsPerMeta );
  
  while ( group_idx_cyclic < m_GroupCount )
  {
    uint32_t group_idx = group_idx_cyclic % m_GroupCount;
    Group*   meta      = &m_Meta[ group_idx ];
    // If anything has been set here before, it's a tombstone sentinel whether or not the empty bit is set
    __m512i meta_data      = _mm512_loadu_epi32( meta );
    uint16_t used_bitset   = _mm512_cmp_epi32_mask( meta_data, _mm512_setzero_si512(), _MM_CMPINT_NE );

    for ( uint32_t i_ctrl = ctrl_idx; i_ctrl < 16; ++i_ctrl)
    {
      // This slot is not empty
      if ( _bextr_u32( used_bitset, i_ctrl, 1 ) )
      {
        // Last 7 bits of hash match, very potentially a match. Gonna do a cache invalidate here to check
        if ( meta->m_Control[ i_ctrl ] == ctrl_cmp )
        {
          Elem* elem = &m_Elems[ group_idx * kElemsPerMeta + i_ctrl ];
          if ( elem->m_Key == key )
          { 
            // We found it, remove it!
            meta->m_Control[ i_ctrl ] &= ~kIsNotEmpty;
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
bool HashMap64< V >::Contains( uint64_t key ) const
{
  return At( key ) != nullptr;
}

//---------------------------------------------------------------------------------
template< typename V >
bool HashMap64< V >::IsFull() const
{
  uint32_t count = 0;
  for ( uint32_t i_grp = 0; i_grp < m_GroupCount; ++i_grp )
  {
    __m512i used_bit       = _mm512_set1_epi32 ( kIsNotEmpty );
    __m512i meta_data      = _mm512_loadu_epi32( &m_Meta[ i_grp ] );
    __m512i used_meta      = _mm512_and_epi32  ( meta_data, used_bit );
    uint16_t used_bitset   = _mm512_cmp_epi32_mask( used_meta, used_bit, _MM_CMPINT_EQ );

    if ( used_bitset != 0xffff )
    {
      return false;
    }
  }

  return true;
}

//---------------------------------------------------------------------------------
template< typename V >
bool HashMap64< V >::IsEmpty() const
{
  for ( uint32_t i_grp = 0; i_grp < m_GroupCount; ++i_grp )
  {
    __m512i used_bit       = _mm512_set1_epi32 ( kIsNotEmpty );
    __m512i meta_data      = _mm512_loadu_epi32( &m_Meta[ i_grp ] );
    __m512i used_meta      = _mm512_and_epi32  ( meta_data, used_bit );
    uint16_t used_bitset   = _mm512_cmp_epi32_mask( used_meta, used_bit, _MM_CMPINT_EQ );

    if ( used_bitset > 0 )
    {
      return false;
    }
  }

  return true;
}

//---------------------------------------------------------------------------------
template< typename V >
uint64_t HashMap64< V >::GetFirstKey()
{
  for ( uint32_t i_grp = 0; i_grp < m_GroupCount; ++i_grp )
  {
    __m512i used_bit       = _mm512_set1_epi32 ( kIsNotEmpty );
    __m512i meta_data      = _mm512_loadu_epi32( &m_Meta[ i_grp ] );
    __m512i used_meta      = _mm512_and_epi32  ( meta_data, used_bit );
    uint16_t used_bitset   = _mm512_cmp_epi32_mask( used_meta, used_bit, _MM_CMPINT_EQ );

    if ( used_bitset > 0 )
    {
      uint32_t first_idx = std::countr_zero( used_bitset );
      return m_Elems[ i_grp * kElemsPerMeta + first_idx ].m_Key;
    }
  }

  return 0;
}

//---------------------------------------------------------------------------------
template< typename V >
bool HashMap64< V >::IsInitialized() const
{
  return m_Backing != nullptr;
}

//---------------------------------------------------------------------------------
template< typename V >
inline typename HashMap64<V>::ControlT HashMap64<V>::CtrlFor( uint64_t key )
{
  return ((uint32_t )key) & kHashRem;
}
