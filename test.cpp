#include "pch.h"

#include "HashMap.h"

static constexpr uint32_t kHashMapSize = 1024;
using HashMapT = HashMap64<uint64_t>;
char kHashMapBacking[ HashMapT::GetRequiredBackingSize( kHashMapSize ) ];

//---------------------------------------------------------------------------------
TEST( Basic, InitMap )
{
  HashMapT hash_map;
  hash_map.InitWithBacking( kHashMapBacking, kHashMapSize );
  EXPECT_EQ( hash_map.GetCapacity(), kHashMapSize );

  hash_map.Destroy();
}

//---------------------------------------------------------------------------------
TEST( Basic, InsertElem )
{
  HashMapT hash_map;
  hash_map.InitWithBacking( kHashMapBacking, kHashMapSize );

  ASSERT_TRUE( hash_map.IsEmpty() );

  static constexpr uint64_t kKey = 0xdeadbeefbaadf00d;
  static constexpr uint64_t kVal = 0x1337c0d3cafebab3;

  {
    uint64_t* v = hash_map.Insert( kKey );
    *v = kVal;
  }

  {
    uint64_t* v = hash_map.At( kKey );
    ASSERT_NE( v, nullptr ) << "Did not find element in hashtable after inserting";
    EXPECT_EQ( *hash_map.At( kKey ), kVal ) << "Value did not match after retrieving";
  }

  {
    uint64_t* nv = hash_map.At( 0xe );
    ASSERT_EQ( nv, nullptr ) << "Found an element when we shouldn't have";
  }

  ASSERT_FALSE( hash_map.IsEmpty() );
  ASSERT_EQ( hash_map.GetCount(), 1 );

  hash_map.Destroy();
}

//---------------------------------------------------------------------------------
TEST( Basic, InsertElems )
{
  HashMapT hash_map;
  hash_map.InitWithBacking( kHashMapBacking, kHashMapSize );

  for ( uint32_t i_val = 0; i_val < 20; ++i_val )
  {
    hash_map.Insert( 1 + i_val, i_val );
  }

  ASSERT_EQ( hash_map.GetCount(), 20 );

  for ( int i_val = 19; i_val >= 0; --i_val )
  {
    ASSERT_TRUE( hash_map.Contains( 1 + i_val ) );

    uint64_t* v = hash_map.At( 1 + i_val );
    ASSERT_NE( v, nullptr ) << "Expected " << i_val;
    ASSERT_EQ( *v, i_val );
  }

  hash_map.Destroy();
}

//---------------------------------------------------------------------------------
TEST( Basic, RemoveElems )
{
  HashMapT hash_map;
  hash_map.InitWithBacking( kHashMapBacking, kHashMapSize );

  for ( uint32_t i_val = 0; i_val < 20; ++i_val )
  {
    hash_map.Insert( 1 + i_val, i_val );

    if ( i_val == 0 || i_val == 3 || i_val == 4 || i_val == 9 )
    {
      hash_map.Remove( 1 + i_val );
    }
  }

  for ( int i_val = 19; i_val >= 0; --i_val )
  {
    uint64_t* v = hash_map.At( 1 + i_val );
    if ( i_val == 0 || i_val == 3 || i_val == 4 || i_val == 9 )
    {
      ASSERT_FALSE( hash_map.Contains( 1 + i_val ) );
      ASSERT_EQ( v, nullptr );
    }
    else
    {
      ASSERT_TRUE( hash_map.Contains( 1 + i_val ) );
      ASSERT_NE( v, nullptr ) << "Expected " << i_val;
      ASSERT_EQ( *v, i_val );
    }
  }

  hash_map.Destroy();
}

//---------------------------------------------------------------------------------
TEST( Basic, Collision )
{
  HashMapT hash_map;
  hash_map.InitWithBacking( kHashMapBacking, kHashMapSize );

  hash_map.Insert( 1, 0xabcd );
  hash_map.Insert( 2, 0xef01 );
  hash_map.Insert( kHashMapSize + 1, 0x01010101 );

  ASSERT_TRUE( hash_map.Contains( kHashMapSize + 1 ) );
  ASSERT_TRUE( hash_map.Contains( 1 ) );

  hash_map.Destroy();
}

//---------------------------------------------------------------------------------
TEST( Basic, DuplicateKeys )
{
  HashMapT hash_map;
  hash_map.InitWithBacking( kHashMapBacking, kHashMapSize );

  hash_map.Insert( 1, 0xabcd );

  ASSERT_EQ( hash_map.GetCount(), 1 );

  hash_map.Insert( 1, 0xf00d );

  ASSERT_EQ( hash_map.GetCount(), 1 );

  hash_map.Destroy();
}

//---------------------------------------------------------------------------------
TEST( Basic, RemoveMissingKey )
{
  HashMapT hash_map;
  hash_map.InitWithBacking( kHashMapBacking, kHashMapSize );

  hash_map.Remove( 1 );

  ASSERT_EQ( hash_map.GetCount(), 0 );

  hash_map.Destroy();
}

//---------------------------------------------------------------------------------
TEST( Basic, LotsOfInserts )
{
  static constexpr uint32_t kHugeHashMapSize = 1048576;

  void* backing = malloc( HashMapT::GetRequiredBackingSize( kHugeHashMapSize ) );

  HashMapT hash_map;
  hash_map.InitWithBacking( backing, kHugeHashMapSize);

  for ( uint32_t i_val = 0; i_val < kHugeHashMapSize - 1; ++i_val )
  {
    hash_map.Insert( 1 + i_val, i_val );
  }

  EXPECT_EQ( hash_map.GetCount(), kHugeHashMapSize - 1 );

  hash_map.Destroy();
  free( backing );
}