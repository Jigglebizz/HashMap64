#pragma once

#include <stdint.h>
#include <intrin.h>
#include "HashMapImpl.h"
#include "HashMapAVX512.h"

#ifndef CORE_API
#define CORE_API
#endif

//---------------------------------------------------------------------------------
template< typename V >
class HashMap64
{
public:
  HashMap64<V>()
  : m_Impl      ( nullptr )
  , m_Backing   ( nullptr )
  #ifdef USE_HEAP
  , m_OwningHeap( nullptr )
  #endif
  {}

  static constexpr size_t   CORE_API GetRequiredBackingSize ( uint32_t capacity );
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
                   bool     CORE_API IsFull                 ( ) const;
                   bool     CORE_API IsEmpty                ( ) const;
                   bool     CORE_API Contains               ( uint64_t key ) const;
                   uint64_t CORE_API GetFirstKey            ( );
  
                   bool     CORE_API IsInitialized          ( ) const;

private:
  HashMap64ImplBase<V>* m_Impl;
  void*                 m_Backing;
  #ifdef USE_HEAP
    MemAllocHeap* m_OwningHeap;
  #endif
};

//---------------------------------------------------------------------------------
class VectorizationCapability
{
public:
  enum Support
  {
    kSupportAvx512,
    kSupportAvx2,

    kVectorizationUnsupported
  };

  VectorizationCapability();

  Support GetSupport() const;
private:
  Support m_Support;
};

//---------------------------------------------------------------------------------
extern const VectorizationCapability s_VectorizationCapabilities;

//---------------------------------------------------------------------------------
template< typename V >
constexpr size_t HashMap64< V >::GetRequiredBackingSize( uint32_t capacity )
{
  return sizeof( HashMap64ImplBase< V > ) + HashMap64ImplBase< V >::GetRequiredBackingSize( capacity );
}

//---------------------------------------------------------------------------------
template< typename V >
void HashMap64< V >::InitWithBacking( void* backing, uint32_t capacity )
{
  m_Backing = backing;
  if ( s_VectorizationCapabilities.GetSupport() == VectorizationCapability::kSupportAvx512 )
  {
    using HashMapT = HashMap64AVX512< V >;
    m_Impl  = new (backing) HashMapT();
    backing = (void*)(((uint8_t*)backing) + sizeof( HashMapT ));
  }
  m_Impl->InitWithBacking( backing, capacity );
}

#ifdef USE_HEAP
//---------------------------------------------------------------------------------
template< typename V >
void HashMap64< V >::InitOnHeap( MemAllocHeap* heap, uint32_t desired_capacity )
{
  // Size must be aligned to 16 elems
  if ( uint32_t rem = desired_capacity % 16 )
  {
    desired_capacity += 16 - rem;
  }

  size_t sz = GetRequiredBackingSize( desired_capacity );
  m_Backing = heap->Alloc( sz );
  ASSERT_MSG( m_Backing != nullptr, "Not enough space in heap for hashmap!" );

  InitWithBacking( m_Backing, desired_capacity );

  m_OwningHeap = heap;
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
bool HashMap64< V >::IsInitialized() const
{
  return m_Backing != nullptr;
}

//---------------------------------------------------------------------------------
template< typename V >
void HashMap64< V >::Insert( uint64_t key, const V& value )
{
  ASSERT_MSG( m_Backing != nullptr, "HashMap not initialized!" );
  m_Impl->Insert( key, value );
}

//---------------------------------------------------------------------------------
template< typename V >
V* HashMap64< V >::Insert( uint64_t key )
{
  ASSERT_MSG( m_Backing != nullptr, "HashMap not initialized!" );
  return m_Impl->Insert( key );
}

//---------------------------------------------------------------------------------
template< typename V >
V* HashMap64< V >::At( uint64_t key )
{
  ASSERT_MSG( m_Backing != nullptr, "HashMap not initialized!" );
  return m_Impl->At( key );
}

//---------------------------------------------------------------------------------
template< typename V >
const V* HashMap64< V >::At( uint64_t key ) const
{
  ASSERT_MSG( m_Backing != nullptr, "HashMap not initialized!" );
  return m_Impl->At( key );
}

//---------------------------------------------------------------------------------
template< typename V >
void HashMap64< V >::Remove( uint64_t key )
{
  ASSERT_MSG( m_Backing != nullptr, "HashMap not initialized!" );
  m_Impl->Remove( key );
}

//---------------------------------------------------------------------------------
template< typename V >
uint32_t HashMap64< V >::GetCapacity( ) const
{
  ASSERT_MSG( m_Backing != nullptr, "HashMap not initialized!" );
  return m_Impl->GetCapacity();
}

//---------------------------------------------------------------------------------
template< typename V >
uint32_t HashMap64< V >::GetCount( ) const
{
  ASSERT_MSG( m_Backing != nullptr, "HashMap not initialized!" );
  return m_Impl->GetCount();
}

//---------------------------------------------------------------------------------
template< typename V >
bool HashMap64< V >::IsFull( ) const
{
  ASSERT_MSG( m_Backing != nullptr, "HashMap not initialized!" );
  return m_Impl->IsFull();
}

//---------------------------------------------------------------------------------
template< typename V >
bool HashMap64< V >::IsEmpty( ) const
{
  ASSERT_MSG( m_Backing != nullptr, "HashMap not initialized!" );
  return m_Impl->IsEmpty();
}

//---------------------------------------------------------------------------------
template< typename V >
bool HashMap64< V >::Contains( uint64_t key ) const
{
  ASSERT_MSG( m_Backing != nullptr, "HashMap not initialized!" );
  return m_Impl->Contains( key );
}

//---------------------------------------------------------------------------------
template< typename V >
uint64_t HashMap64< V >::GetFirstKey( )
{
  ASSERT_MSG( m_Backing != nullptr, "HashMap not initialized!" );
  return m_Impl->GetFirstKey( );
}