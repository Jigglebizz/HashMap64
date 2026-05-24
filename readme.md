This is an AVX-512 driven Hashmap and the accompanying unit test

It employs dynamic dispatch to also provide AVX2 support, though it is ~2x slower

It's similar to Google's Swiss Hash Table