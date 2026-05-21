#include "HashMap.h"


//---------------------------------------------------------------------------------
//
// Vectorization support determination
// 
//---------------------------------------------------------------------------------
const VectorizationCapability s_VectorizationCapabilities;

//---------------------------------------------------------------------------------
VectorizationCapability::VectorizationCapability()
{
  // EAX, EBX, ECX, EDX in order
  int cpu_info[4];
  __cpuid( cpu_info, 7 );

  enum CpuCapabilitiesFlags
  {
    kCapabilityEbxAvx512F = ( 1 << 16 ),
    kCapabilityEbxAvx2    = ( 1 << 5  )
  };

  if ( cpu_info[1] & kCapabilityEbxAvx512F )
  {
    m_Support = kSupportAvx512;
  }
  else if ( cpu_info[1] & kCapabilityEbxAvx2 )
  {
    m_Support = kSupportAvx2;
  }
}

//---------------------------------------------------------------------------------
VectorizationCapability::Support VectorizationCapability::GetSupport() const
{
  return m_Support;
}