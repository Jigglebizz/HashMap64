This is an AVX-512 driven Hashmap and the accompanying unit test

It employs dynamic dispatch to also provide AVX2 support. It has nearly identical performance under normal circumstances, but is nearly ~2x slower under high load

It's similar to Google's Swiss Hash Table