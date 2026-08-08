// Shortest float64 formatting: Schubfach, with the render attached.
//
// A transliteration of simdjson's Go implementation (schubfach.go,
// pow10_table.go, the appendShortest renderer and appendFloat's format
// rule), which is itself checked against strconv on every value its tests
// and fuzzers reach. The contract here is exactly that code's output for
// a finite float64: encoding/json's format rule (decimal in [1e-6, 1e21),
// scientific outside, single-digit exponents unpadded), the whole-number
// fast path included, negative zero as "-0". The reference in
// internal/ref is the same transliteration in Go; where the three
// disagree, the two copies here are wrong.
//
// dst must have room for 25 bytes: the longest shortest-form float64 is
// 24 characters and the sign is counted. The guard enforces it.

#include "goabi.h"

typedef long isize;
typedef long long i64;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef double f64;
typedef unsigned __int128 u128;

#define POW10_MIN (-348)

static const u64 dt_pow10_hi[] = {
  0xfa8fd5a0081c0288ull, 0x9c99e58405118195ull, 0xc3c05ee50655e1faull, 
  0xf4b0769e47eb5a78ull, 0x98ee4a22ecf3188bull, 0xbf29dcaba82fdeaeull, 
  0xeef453d6923bd65aull, 0x9558b4661b6565f8ull, 0xbaaee17fa23ebf76ull, 
  0xe95a99df8ace6f53ull, 0x91d8a02bb6c10594ull, 0xb64ec836a47146f9ull, 
  0xe3e27a444d8d98b7ull, 0x8e6d8c6ab0787f72ull, 0xb208ef855c969f4full, 
  0xde8b2b66b3bc4723ull, 0x8b16fb203055ac76ull, 0xaddcb9e83c6b1793ull, 
  0xd953e8624b85dd78ull, 0x87d4713d6f33aa6bull, 0xa9c98d8ccb009506ull, 
  0xd43bf0effdc0ba48ull, 0x84a57695fe98746dull, 0xa5ced43b7e3e9188ull, 
  0xcf42894a5dce35eaull, 0x818995ce7aa0e1b2ull, 0xa1ebfb4219491a1full, 
  0xca66fa129f9b60a6ull, 0xfd00b897478238d0ull, 0x9e20735e8cb16382ull, 
  0xc5a890362fddbc62ull, 0xf712b443bbd52b7bull, 0x9a6bb0aa55653b2dull, 
  0xc1069cd4eabe89f8ull, 0xf148440a256e2c76ull, 0x96cd2a865764dbcaull, 
  0xbc807527ed3e12bcull, 0xeba09271e88d976bull, 0x93445b8731587ea3ull, 
  0xb8157268fdae9e4cull, 0xe61acf033d1a45dfull, 0x8fd0c16206306babull, 
  0xb3c4f1ba87bc8696ull, 0xe0b62e2929aba83cull, 0x8c71dcd9ba0b4925ull, 
  0xaf8e5410288e1b6full, 0xdb71e91432b1a24aull, 0x892731ac9faf056eull, 
  0xab70fe17c79ac6caull, 0xd64d3d9db981787dull, 0x85f0468293f0eb4eull, 
  0xa76c582338ed2621ull, 0xd1476e2c07286faaull, 0x82cca4db847945caull, 
  0xa37fce126597973cull, 0xcc5fc196fefd7d0cull, 0xff77b1fcbebcdc4full, 
  0x9faacf3df73609b1ull, 0xc795830d75038c1dull, 0xf97ae3d0d2446f25ull, 
  0x9becce62836ac577ull, 0xc2e801fb244576d5ull, 0xf3a20279ed56d48aull, 
  0x9845418c345644d6ull, 0xbe5691ef416bd60cull, 0xedec366b11c6cb8full, 
  0x94b3a202eb1c3f39ull, 0xb9e08a83a5e34f07ull, 0xe858ad248f5c22c9ull, 
  0x91376c36d99995beull, 0xb58547448ffffb2dull, 0xe2e69915b3fff9f9ull, 
  0x8dd01fad907ffc3bull, 0xb1442798f49ffb4aull, 0xdd95317f31c7fa1dull, 
  0x8a7d3eef7f1cfc52ull, 0xad1c8eab5ee43b66ull, 0xd863b256369d4a40ull, 
  0x873e4f75e2224e68ull, 0xa90de3535aaae202ull, 0xd3515c2831559a83ull, 
  0x8412d9991ed58091ull, 0xa5178fff668ae0b6ull, 0xce5d73ff402d98e3ull, 
  0x80fa687f881c7f8eull, 0xa139029f6a239f72ull, 0xc987434744ac874eull, 
  0xfbe9141915d7a922ull, 0x9d71ac8fada6c9b5ull, 0xc4ce17b399107c22ull, 
  0xf6019da07f549b2bull, 0x99c102844f94e0fbull, 0xc0314325637a1939ull, 
  0xf03d93eebc589f88ull, 0x96267c7535b763b5ull, 0xbbb01b9283253ca2ull, 
  0xea9c227723ee8bcbull, 0x92a1958a7675175full, 0xb749faed14125d36ull, 
  0xe51c79a85916f484ull, 0x8f31cc0937ae58d2ull, 0xb2fe3f0b8599ef07ull, 
  0xdfbdcece67006ac9ull, 0x8bd6a141006042bdull, 0xaecc49914078536dull, 
  0xda7f5bf590966848ull, 0x888f99797a5e012dull, 0xaab37fd7d8f58178ull, 
  0xd5605fcdcf32e1d6ull, 0x855c3be0a17fcd26ull, 0xa6b34ad8c9dfc06full, 
  0xd0601d8efc57b08bull, 0x823c12795db6ce57ull, 0xa2cb1717b52481edull, 
  0xcb7ddcdda26da268ull, 0xfe5d54150b090b02ull, 0x9efa548d26e5a6e1ull, 
  0xc6b8e9b0709f109aull, 0xf867241c8cc6d4c0ull, 0x9b407691d7fc44f8ull, 
  0xc21094364dfb5636ull, 0xf294b943e17a2bc4ull, 0x979cf3ca6cec5b5aull, 
  0xbd8430bd08277231ull, 0xece53cec4a314ebdull, 0x940f4613ae5ed136ull, 
  0xb913179899f68584ull, 0xe757dd7ec07426e5ull, 0x9096ea6f3848984full, 
  0xb4bca50b065abe63ull, 0xe1ebce4dc7f16dfbull, 0x8d3360f09cf6e4bdull, 
  0xb080392cc4349decull, 0xdca04777f541c567ull, 0x89e42caaf9491b60ull, 
  0xac5d37d5b79b6239ull, 0xd77485cb25823ac7ull, 0x86a8d39ef77164bcull, 
  0xa8530886b54dbdebull, 0xd267caa862a12d66ull, 0x8380dea93da4bc60ull, 
  0xa46116538d0deb78ull, 0xcd795be870516656ull, 0x806bd9714632dff6ull, 
  0xa086cfcd97bf97f3ull, 0xc8a883c0fdaf7df0ull, 0xfad2a4b13d1b5d6cull, 
  0x9cc3a6eec6311a63ull, 0xc3f490aa77bd60fcull, 0xf4f1b4d515acb93bull, 
  0x991711052d8bf3c5ull, 0xbf5cd54678eef0b6ull, 0xef340a98172aace4ull, 
  0x9580869f0e7aac0eull, 0xbae0a846d2195712ull, 0xe998d258869facd7ull, 
  0x91ff83775423cc06ull, 0xb67f6455292cbf08ull, 0xe41f3d6a7377eecaull, 
  0x8e938662882af53eull, 0xb23867fb2a35b28dull, 0xdec681f9f4c31f31ull, 
  0x8b3c113c38f9f37eull, 0xae0b158b4738705eull, 0xd98ddaee19068c76ull, 
  0x87f8a8d4cfa417c9ull, 0xa9f6d30a038d1dbcull, 0xd47487cc8470652bull, 
  0x84c8d4dfd2c63f3bull, 0xa5fb0a17c777cf09ull, 0xcf79cc9db955c2ccull, 
  0x81ac1fe293d599bfull, 0xa21727db38cb002full, 0xca9cf1d206fdc03bull, 
  0xfd442e4688bd304aull, 0x9e4a9cec15763e2eull, 0xc5dd44271ad3cdbaull, 
  0xf7549530e188c128ull, 0x9a94dd3e8cf578b9ull, 0xc13a148e3032d6e7ull, 
  0xf18899b1bc3f8ca1ull, 0x96f5600f15a7b7e5ull, 0xbcb2b812db11a5deull, 
  0xebdf661791d60f56ull, 0x936b9fcebb25c995ull, 0xb84687c269ef3bfbull, 
  0xe65829b3046b0afaull, 0x8ff71a0fe2c2e6dcull, 0xb3f4e093db73a093ull, 
  0xe0f218b8d25088b8ull, 0x8c974f7383725573ull, 0xafbd2350644eeacfull, 
  0xdbac6c247d62a583ull, 0x894bc396ce5da772ull, 0xab9eb47c81f5114full, 
  0xd686619ba27255a2ull, 0x8613fd0145877585ull, 0xa798fc4196e952e7ull, 
  0xd17f3b51fca3a7a0ull, 0x82ef85133de648c4ull, 0xa3ab66580d5fdaf5ull, 
  0xcc963fee10b7d1b3ull, 0xffbbcfe994e5c61full, 0x9fd561f1fd0f9bd3ull, 
  0xc7caba6e7c5382c8ull, 0xf9bd690a1b68637bull, 0x9c1661a651213e2dull, 
  0xc31bfa0fe5698db8ull, 0xf3e2f893dec3f126ull, 0x986ddb5c6b3a76b7ull, 
  0xbe89523386091465ull, 0xee2ba6c0678b597full, 0x94db483840b717efull, 
  0xba121a4650e4ddebull, 0xe896a0d7e51e1566ull, 0x915e2486ef32cd60ull, 
  0xb5b5ada8aaff80b8ull, 0xe3231912d5bf60e6ull, 0x8df5efabc5979c8full, 
  0xb1736b96b6fd83b3ull, 0xddd0467c64bce4a0ull, 0x8aa22c0dbef60ee4ull, 
  0xad4ab7112eb3929dull, 0xd89d64d57a607744ull, 0x87625f056c7c4a8bull, 
  0xa93af6c6c79b5d2dull, 0xd389b47879823479ull, 0x843610cb4bf160cbull, 
  0xa54394fe1eedb8feull, 0xce947a3da6a9273eull, 0x811ccc668829b887ull, 
  0xa163ff802a3426a8ull, 0xc9bcff6034c13052ull, 0xfc2c3f3841f17c67ull, 
  0x9d9ba7832936edc0ull, 0xc5029163f384a931ull, 0xf64335bcf065d37dull, 
  0x99ea0196163fa42eull, 0xc06481fb9bcf8d39ull, 0xf07da27a82c37088ull, 
  0x964e858c91ba2655ull, 0xbbe226efb628afeaull, 0xeadab0aba3b2dbe5ull, 
  0x92c8ae6b464fc96full, 0xb77ada0617e3bbcbull, 0xe55990879ddcaabdull, 
  0x8f57fa54c2a9eab6ull, 0xb32df8e9f3546564ull, 0xdff9772470297ebdull, 
  0x8bfbea76c619ef36ull, 0xaefae51477a06b03ull, 0xdab99e59958885c4ull, 
  0x88b402f7fd75539bull, 0xaae103b5fcd2a881ull, 0xd59944a37c0752a2ull, 
  0x857fcae62d8493a5ull, 0xa6dfbd9fb8e5b88eull, 0xd097ad07a71f26b2ull, 
  0x825ecc24c873782full, 0xa2f67f2dfa90563bull, 0xcbb41ef979346bcaull, 
  0xfea126b7d78186bcull, 0x9f24b832e6b0f436ull, 0xc6ede63fa05d3143ull, 
  0xf8a95fcf88747d94ull, 0x9b69dbe1b548ce7cull, 0xc24452da229b021bull, 
  0xf2d56790ab41c2a2ull, 0x97c560ba6b0919a5ull, 0xbdb6b8e905cb600full, 
  0xed246723473e3813ull, 0x9436c0760c86e30bull, 0xb94470938fa89bceull, 
  0xe7958cb87392c2c2ull, 0x90bd77f3483bb9b9ull, 0xb4ecd5f01a4aa828ull, 
  0xe2280b6c20dd5232ull, 0x8d590723948a535full, 0xb0af48ec79ace837ull, 
  0xdcdb1b2798182244ull, 0x8a08f0f8bf0f156bull, 0xac8b2d36eed2dac5ull, 
  0xd7adf884aa879177ull, 0x86ccbb52ea94baeaull, 0xa87fea27a539e9a5ull, 
  0xd29fe4b18e88640eull, 0x83a3eeeef9153e89ull, 0xa48ceaaab75a8e2bull, 
  0xcdb02555653131b6ull, 0x808e17555f3ebf11ull, 0xa0b19d2ab70e6ed6ull, 
  0xc8de047564d20a8bull, 0xfb158592be068d2eull, 0x9ced737bb6c4183dull, 
  0xc428d05aa4751e4cull, 0xf53304714d9265dfull, 0x993fe2c6d07b7fabull, 
  0xbf8fdb78849a5f96ull, 0xef73d256a5c0f77cull, 0x95a8637627989aadull, 
  0xbb127c53b17ec159ull, 0xe9d71b689dde71afull, 0x9226712162ab070dull, 
  0xb6b00d69bb55c8d1ull, 0xe45c10c42a2b3b05ull, 0x8eb98a7a9a5b04e3ull, 
  0xb267ed1940f1c61cull, 0xdf01e85f912e37a3ull, 0x8b61313bbabce2c6ull, 
  0xae397d8aa96c1b77ull, 0xd9c7dced53c72255ull, 0x881cea14545c7575ull, 
  0xaa242499697392d2ull, 0xd4ad2dbfc3d07787ull, 0x84ec3c97da624ab4ull, 
  0xa6274bbdd0fadd61ull, 0xcfb11ead453994baull, 0x81ceb32c4b43fcf4ull, 
  0xa2425ff75e14fc31ull, 0xcad2f7f5359a3b3eull, 0xfd87b5f28300ca0dull, 
  0x9e74d1b791e07e48ull, 0xc612062576589ddaull, 0xf79687aed3eec551ull, 
  0x9abe14cd44753b52ull, 0xc16d9a0095928a27ull, 0xf1c90080baf72cb1ull, 
  0x971da05074da7beeull, 0xbce5086492111aeaull, 0xec1e4a7db69561a5ull, 
  0x9392ee8e921d5d07ull, 0xb877aa3236a4b449ull, 0xe69594bec44de15bull, 
  0x901d7cf73ab0acd9ull, 0xb424dc35095cd80full, 0xe12e13424bb40e13ull, 
  0x8cbccc096f5088cbull, 0xafebff0bcb24aafeull, 0xdbe6fecebdedd5beull, 
  0x89705f4136b4a597ull, 0xabcc77118461cefcull, 0xd6bf94d5e57a42bcull, 
  0x8637bd05af6c69b5ull, 0xa7c5ac471b478423ull, 0xd1b71758e219652bull, 
  0x83126e978d4fdf3bull, 0xa3d70a3d70a3d70aull, 0xccccccccccccccccull, 
  0x8000000000000000ull, 0xa000000000000000ull, 0xc800000000000000ull, 
  0xfa00000000000000ull, 0x9c40000000000000ull, 0xc350000000000000ull, 
  0xf424000000000000ull, 0x9896800000000000ull, 0xbebc200000000000ull, 
  0xee6b280000000000ull, 0x9502f90000000000ull, 0xba43b74000000000ull, 
  0xe8d4a51000000000ull, 0x9184e72a00000000ull, 0xb5e620f480000000ull, 
  0xe35fa931a0000000ull, 0x8e1bc9bf04000000ull, 0xb1a2bc2ec5000000ull, 
  0xde0b6b3a76400000ull, 0x8ac7230489e80000ull, 0xad78ebc5ac620000ull, 
  0xd8d726b7177a8000ull, 0x878678326eac9000ull, 0xa968163f0a57b400ull, 
  0xd3c21bcecceda100ull, 0x84595161401484a0ull, 0xa56fa5b99019a5c8ull, 
  0xcecb8f27f4200f3aull, 0x813f3978f8940984ull, 0xa18f07d736b90be5ull, 
  0xc9f2c9cd04674edeull, 0xfc6f7c4045812296ull, 0x9dc5ada82b70b59dull, 
  0xc5371912364ce305ull, 0xf684df56c3e01bc6ull, 0x9a130b963a6c115cull, 
  0xc097ce7bc90715b3ull, 0xf0bdc21abb48db20ull, 0x96769950b50d88f4ull, 
  0xbc143fa4e250eb31ull, 0xeb194f8e1ae525fdull, 0x92efd1b8d0cf37beull, 
  0xb7abc627050305adull, 0xe596b7b0c643c719ull, 0x8f7e32ce7bea5c6full, 
  0xb35dbf821ae4f38bull, 0xe0352f62a19e306eull, 0x8c213d9da502de45ull, 
  0xaf298d050e4395d6ull, 0xdaf3f04651d47b4cull, 0x88d8762bf324cd0full, 
  0xab0e93b6efee0053ull, 0xd5d238a4abe98068ull, 0x85a36366eb71f041ull, 
  0xa70c3c40a64e6c51ull, 0xd0cf4b50cfe20765ull, 0x82818f1281ed449full, 
  0xa321f2d7226895c7ull, 0xcbea6f8ceb02bb39ull, 0xfee50b7025c36a08ull, 
  0x9f4f2726179a2245ull, 0xc722f0ef9d80aad6ull, 0xf8ebad2b84e0d58bull, 
  0x9b934c3b330c8577ull, 0xc2781f49ffcfa6d5ull, 0xf316271c7fc3908aull, 
  0x97edd871cfda3a56ull, 0xbde94e8e43d0c8ecull, 0xed63a231d4c4fb27ull, 
  0x945e455f24fb1cf8ull, 0xb975d6b6ee39e436ull, 0xe7d34c64a9c85d44ull, 
  0x90e40fbeea1d3a4aull, 0xb51d13aea4a488ddull, 0xe264589a4dcdab14ull, 
  0x8d7eb76070a08aecull, 0xb0de65388cc8ada8ull, 0xdd15fe86affad912ull, 
  0x8a2dbf142dfcc7abull, 0xacb92ed9397bf996ull, 0xd7e77a8f87daf7fbull, 
  0x86f0ac99b4e8dafdull, 0xa8acd7c0222311bcull, 0xd2d80db02aabd62bull, 
  0x83c7088e1aab65dbull, 0xa4b8cab1a1563f52ull, 0xcde6fd5e09abcf26ull, 
  0x80b05e5ac60b6178ull, 0xa0dc75f1778e39d6ull, 0xc913936dd571c84cull, 
  0xfb5878494ace3a5full, 0x9d174b2dcec0e47bull, 0xc45d1df942711d9aull, 
  0xf5746577930d6500ull, 0x9968bf6abbe85f20ull, 0xbfc2ef456ae276e8ull, 
  0xefb3ab16c59b14a2ull, 0x95d04aee3b80ece5ull, 0xbb445da9ca61281full, 
  0xea1575143cf97226ull, 0x924d692ca61be758ull, 0xb6e0c377cfa2e12eull, 
  0xe498f455c38b997aull, 0x8edf98b59a373fecull, 0xb2977ee300c50fe7ull, 
  0xdf3d5e9bc0f653e1ull, 0x8b865b215899f46cull, 0xae67f1e9aec07187ull, 
  0xda01ee641a708de9ull, 0x884134fe908658b2ull, 0xaa51823e34a7eedeull, 
  0xd4e5e2cdc1d1ea96ull, 0x850fadc09923329eull, 0xa6539930bf6bff45ull, 
  0xcfe87f7cef46ff16ull, 0x81f14fae158c5f6eull, 0xa26da3999aef7749ull, 
  0xcb090c8001ab551cull, 0xfdcb4fa002162a63ull, 0x9e9f11c4014dda7eull, 
  0xc646d63501a1511dull, 0xf7d88bc24209a565ull, 0x9ae757596946075full, 
  0xc1a12d2fc3978937ull, 0xf209787bb47d6b84ull, 0x9745eb4d50ce6332ull, 
  0xbd176620a501fbffull, 0xec5d3fa8ce427affull, 0x93ba47c980e98cdfull, 
  0xb8a8d9bbe123f017ull, 0xe6d3102ad96cec1dull, 0x9043ea1ac7e41392ull, 
  0xb454e4a179dd1877ull, 0xe16a1dc9d8545e94ull, 0x8ce2529e2734bb1dull, 
  0xb01ae745b101e9e4ull, 0xdc21a1171d42645dull, 0x899504ae72497ebaull, 
  0xabfa45da0edbde69ull, 0xd6f8d7509292d603ull, 0x865b86925b9bc5c2ull, 
  0xa7f26836f282b732ull, 0xd1ef0244af2364ffull, 0x8335616aed761f1full, 
  0xa402b9c5a8d3a6e7ull, 0xcd036837130890a1ull, 0x802221226be55a64ull, 
  0xa02aa96b06deb0fdull, 0xc83553c5c8965d3dull, 0xfa42a8b73abbf48cull, 
  0x9c69a97284b578d7ull, 0xc38413cf25e2d70dull, 0xf46518c2ef5b8cd1ull, 
  0x98bf2f79d5993802ull, 0xbeeefb584aff8603ull, 0xeeaaba2e5dbf6784ull, 
  0x952ab45cfa97a0b2ull, 0xba756174393d88dfull, 0xe912b9d1478ceb17ull, 
  0x91abb422ccb812eeull, 0xb616a12b7fe617aaull, 0xe39c49765fdf9d94ull, 
  0x8e41ade9fbebc27dull, 0xb1d219647ae6b31cull, 0xde469fbd99a05fe3ull, 
  0x8aec23d680043beeull, 0xada72ccc20054ae9ull, 0xd910f7ff28069da4ull, 
  0x87aa9aff79042286ull, 0xa99541bf57452b28ull, 0xd3fa922f2d1675f2ull, 
  0x847c9b5d7c2e09b7ull, 0xa59bc234db398c25ull, 0xcf02b2c21207ef2eull, 
  0x8161afb94b44f57dull, 0xa1ba1ba79e1632dcull, 0xca28a291859bbf93ull, 
  0xfcb2cb35e702af78ull, 0x9defbf01b061adabull, 0xc56baec21c7a1916ull, 
  0xf6c69a72a3989f5bull, 0x9a3c2087a63f6399ull, 0xc0cb28a98fcf3c7full, 
  0xf0fdf2d3f3c30b9full, 0x969eb7c47859e743ull, 0xbc4665b596706114ull, 
  0xeb57ff22fc0c7959ull, 0x9316ff75dd87cbd8ull, 0xb7dcbf5354e9beceull, 
  0xe5d3ef282a242e81ull, 0x8fa475791a569d10ull, 0xb38d92d760ec4455ull, 
  0xe070f78d3927556aull, 0x8c469ab843b89562ull, 0xaf58416654a6babbull, 
  0xdb2e51bfe9d0696aull, 0x88fcf317f22241e2ull, 0xab3c2fddeeaad25aull, 
  0xd60b3bd56a5586f1ull, 0x85c7056562757456ull, 0xa738c6bebb12d16cull, 
  0xd106f86e69d785c7ull, 0x82a45b450226b39cull, 0xa34d721642b06084ull, 
  0xcc20ce9bd35c78a5ull, 0xff290242c83396ceull, 0x9f79a169bd203e41ull, 
  0xc75809c42c684dd1ull, 0xf92e0c3537826145ull, 0x9bbcc7a142b17ccbull, 
  0xc2abf989935ddbfeull, 0xf356f7ebf83552feull, 0x98165af37b2153deull, 
  0xbe1bf1b059e9a8d6ull, 0xeda2ee1c7064130cull, 0x9485d4d1c63e8be7ull, 
  0xb9a74a0637ce2ee1ull, 0xe8111c87c5c1ba99ull, 0x910ab1d4db9914a0ull, 
  0xb54d5e4a127f59c8ull, 0xe2a0b5dc971f303aull, 0x8da471a9de737e24ull, 
  0xb10d8e1456105dadull, 0xdd50f1996b947518ull, 0x8a5296ffe33cc92full, 
  0xace73cbfdc0bfb7bull, 0xd8210befd30efa5aull, 0x8714a775e3e95c78ull, 
  0xa8d9d1535ce3b396ull, 0xd31045a8341ca07cull, 0x83ea2b892091e44dull, 
  0xa4e4b66b68b65d60ull, 0xce1de40642e3f4b9ull, 0x80d2ae83e9ce78f3ull, 
  0xa1075a24e4421730ull, 0xc94930ae1d529cfcull, 0xfb9b7cd9a4a7443cull, 
  0x9d412e0806e88aa5ull, 0xc491798a08a2ad4eull, 0xf5b5d7ec8acb58a2ull, 
  0x9991a6f3d6bf1765ull, 0xbff610b0cc6edd3full, 0xeff394dcff8a948eull, 
  0x95f83d0a1fb69cd9ull, 0xbb764c4ca7a4440full, 0xea53df5fd18d5513ull, 
  0x92746b9be2f8552cull, 0xb7118682dbb66a77ull, 0xe4d5e82392a40515ull, 
  0x8f05b1163ba6832dull, 0xb2c71d5bca9023f8ull, 0xdf78e4b2bd342cf6ull, 
  0x8bab8eefb6409c1aull, 0xae9672aba3d0c320ull, 0xda3c0f568cc4f3e8ull, 
  0x8865899617fb1871ull, 0xaa7eebfb9df9de8dull, 0xd51ea6fa85785631ull, 
  0x8533285c936b35deull, 0xa67ff273b8460356ull, 0xd01fef10a657842cull, 
  0x8213f56a67f6b29bull, 0xa298f2c501f45f42ull, 0xcb3f2f7642717713ull, 
  0xfe0efb53d30dd4d7ull, 0x9ec95d1463e8a506ull, 0xc67bb4597ce2ce48ull, 
  0xf81aa16fdc1b81daull, 0x9b10a4e5e9913128ull, 0xc1d4ce1f63f57d72ull, 
  0xf24a01a73cf2dccfull, 0x976e41088617ca01ull, 0xbd49d14aa79dbc82ull, 
  0xec9c459d51852ba2ull, 0x93e1ab8252f33b45ull, 0xb8da1662e7b00a17ull, 
  0xe7109bfba19c0c9dull, 0x906a617d450187e2ull, 0xb484f9dc9641e9daull, 
  0xe1a63853bbd26451ull, 0x8d07e33455637eb2ull, 0xb049dc016abc5e5full, 
  0xdc5c5301c56b75f7ull, 0x89b9b3e11b6329baull, 0xac2820d9623bf429ull, 
  0xd732290fbacaf133ull, 0x867f59a9d4bed6c0ull, 0xa81f301449ee8c70ull, 
  0xd226fc195c6a2f8cull, 0x83585d8fd9c25db7ull, 0xa42e74f3d032f525ull, 
  0xcd3a1230c43fb26full, 0x80444b5e7aa7cf85ull, 0xa0555e361951c366ull, 
  0xc86ab5c39fa63440ull, 0xfa856334878fc150ull, 0x9c935e00d4b9d8d2ull, 
  0xc3b8358109e84f07ull, 0xf4a642e14c6262c8ull, 0x98e7e9cccfbd7dbdull, 
  0xbf21e44003acdd2cull, 0xeeea5d5004981478ull, 0x95527a5202df0ccbull, 
  0xbaa718e68396cffdull, 0xe950df20247c83fdull, 0x91d28b7416cdd27eull, 
  0xb6472e511c81471dull, 0xe3d8f9e563a198e5ull, 0x8e679c2f5e44ff8full, 
  0xb201833b35d63f73ull, 0xde81e40a034bcf4full, 0x8b112e86420f6191ull, 
  0xadd57a27d29339f6ull, 0xd94ad8b1c7380874ull, 0x87cec76f1c830548ull, 
  0xa9c2794ae3a3c69aull, 0xd433179d9c8cb841ull, 0x849feec281d7f328ull, 
  0xa5c7ea73224deff3ull, 0xcf39e50feae16befull, 0x81842f29f2cce375ull, 
  0xa1e53af46f801c53ull, 0xca5e89b18b602368ull, 0xfcf62c1dee382c42ull, 
  0x9e19db92b4e31ba9ull, 0xc5a05277621be293ull, 0xf70867153aa2db38ull, 
  0x9a65406d44a5c903ull, 0xc0fe908895cf3b44ull, 0xf13e34aabb430a15ull, 
  0x96c6e0eab509e64dull, 0xbc789925624c5fe0ull, 0xeb96bf6ebadf77d8ull, 
  0x933e37a534cbaae7ull, 0xb80dc58e81fe95a1ull, 0xe61136f2227e3b09ull, 
  0x8fcac257558ee4e6ull, 0xb3bd72ed2af29e1full, 0xe0accfa875af45a7ull, 
  0x8c6c01c9498d8b88ull, 0xaf87023b9bf0ee6aull, 0xdb68c2ca82ed2a05ull, 
  0x892179be91d43a43ull, 0xab69d82e364948d4ull, 0xd6444e39c3db9b09ull, 
  0x85eab0e41a6940e5ull, 0xa7655d1d2103911full, 0xd13eb46469447567ull, 
  0x82c730bec1cac960ull, 
};

static const u64 dt_pow10_lo[] = {
  0x1732c869cd60e454ull, 0x0e7fbd42205c8eb5ull, 0x521fac92a873b262ull, 
  0xe6a797b752909efaull, 0x9028bed2939a635dull, 0x7432ee873880fc34ull, 
  0x113faa2906a13b40ull, 0x4ac7ca59a424c508ull, 0x5d79bcf00d2df64aull, 
  0xf4d82c2c107973ddull, 0x79071b9b8a4be86aull, 0x9748e2826cdee285ull, 
  0xfd1b1b2308169b26ull, 0xfe30f0f5e50e20f8ull, 0xbdbd2d335e51a936ull, 
  0xad2c788035e61383ull, 0x4c3bcb5021afcc32ull, 0xdf4abe242a1bbf3eull, 
  0xd71d6dad34a2af0eull, 0x8672648c40e5ad69ull, 0x680efdaf511f18c3ull, 
  0x0212bd1b2566def3ull, 0x014bb630f7604b58ull, 0x419ea3bd35385e2eull, 
  0x52064cac828675baull, 0x7343efebd1940994ull, 0x1014ebe6c5f90bf9ull, 
  0xd41a26e077774ef7ull, 0x8920b098955522b5ull, 0x55b46e5f5d5535b1ull, 
  0xeb2189f734aa831eull, 0xa5e9ec7501d523e5ull, 0x47b233c92125366full, 
  0x999ec0bb696e840bull, 0xc00670ea43ca250eull, 0x380406926a5e5729ull, 
  0xc605083704f5ecf3ull, 0xf7864a44c633682full, 0x7ab3ee6afbe0211eull, 
  0x5960ea05bad82965ull, 0x6fb92487298e33beull, 0xa5d3b6d479f8e057ull, 
  0x8f48a4899877186dull, 0x331acdabfe94de88ull, 0x9ff0c08b7f1d0b15ull, 
  0x07ecf0ae5ee44ddaull, 0xc9e82cd9f69d6151ull, 0xbe311c083a225cd3ull, 
  0x6dbd630a48aaf407ull, 0x092cbbccdad5b109ull, 0x25bbf56008c58ea6ull, 
  0xaf2af2b80af6f24full, 0x1af5af660db4aee2ull, 0x50d98d9fc890ed4eull, 
  0xe50ff107bab528a1ull, 0x1e53ed49a96272c9ull, 0x25e8e89c13bb0f7bull, 
  0x77b191618c54e9adull, 0xd59df5b9ef6a2418ull, 0x4b0573286b44ad1eull, 
  0x4ee367f9430aec33ull, 0x229c41f793cda740ull, 0x6b43527578c11110ull, 
  0x830a13896b78aaaaull, 0x23cc986bc656d554ull, 0x2cbfbe86b7ec8aa9ull, 
  0x7bf7d71432f3d6aaull, 0xdaf5ccd93fb0cc54ull, 0xd1b3400f8f9cff69ull, 
  0x23100809b9c21fa2ull, 0xabd40a0c2832a78bull, 0x16c90c8f323f516dull, 
  0xae3da7d97f6792e4ull, 0x99cd11cfdf41779dull, 0x40405643d711d584ull, 
  0x482835ea666b2573ull, 0xda3243650005eed0ull, 0x90bed43e40076a83ull, 
  0x5a7744a6e804a292ull, 0x711515d0a205cb37ull, 0x0d5a5b44ca873e04ull, 
  0xe858790afe9486c3ull, 0x626e974dbe39a873ull, 0xfb0a3d212dc81290ull, 
  0x7ce66634bc9d0b9aull, 0x1c1fffc1ebc44e81ull, 0xa327ffb266b56221ull, 
  0x4bf1ff9f0062baa9ull, 0x6f773fc3603db4aaull, 0xcb550fb4384d21d4ull, 
  0x7e2a53a146606a49ull, 0x2eda7444cbfc426eull, 0xfa911155fefb5309ull, 
  0x793555ab7eba27cbull, 0x4bc1558b2f3458dfull, 0x9eb1aaedfb016f17ull, 
  0x465e15a979c1caddull, 0x0bfacd89ec191ecaull, 0xcef980ec671f667cull, 
  0x82b7e12780e7401bull, 0xd1b2ecb8b0908811ull, 0x861fa7e6dcb4aa16ull, 
  0x67a791e093e1d49bull, 0xe0c8bb2c5c6d24e1ull, 0x58fae9f773886e19ull, 
  0xaf39a475506a899full, 0x6d8406c952429604ull, 0xc8e5087ba6d33b84ull, 
  0xfb1e4a9a90880a65ull, 0x5cf2eea09a550680ull, 0xf42faa48c0ea481full, 
  0xf13b94daf124da27ull, 0x76c53d08d6b70859ull, 0x54768c4b0c64ca6full, 
  0xa9942f5dcf7dfd0aull, 0xd3f93b35435d7c4dull, 0xc47bc5014a1a6db0ull, 
  0x359ab6419ca1091cull, 0xc30163d203c94b63ull, 0x79e0de63425dcf1eull, 
  0x985915fc12f542e5ull, 0x3e6f5b7b17b2939eull, 0xa705992ceecf9c43ull, 
  0x50c6ff782a838354ull, 0xa4f8bf5635246429ull, 0x871b7795e136be9aull, 
  0x28e2557b59846e40ull, 0x331aeada2fe589d0ull, 0x3ff0d2c85def7622ull, 
  0x0fed077a756b53aaull, 0xd3e8495912c62895ull, 0x64712dd7abbbd95dull, 
  0xbd8d794d96aacfb4ull, 0xecf0d7a0fc5583a1ull, 0xf41686c49db57245ull, 
  0x311c2875c522ced6ull, 0x7d633293366b828cull, 0xae5dff9c02033198ull, 
  0xd9f57f830283fdfdull, 0xd072df63c324fd7cull, 0x4247cb9e59f71e6eull, 
  0x52d9be85f074e609ull, 0x67902e276c921f8cull, 0x00ba1cd8a3db53b7ull, 
  0x80e8a40eccd228a5ull, 0x6122cd128006b2ceull, 0x796b805720085f82ull, 
  0xcbe3303674053bb1ull, 0xbedbfc4411068a9dull, 0xee92fb5515482d45ull, 
  0x751bdd152d4d1c4bull, 0xd262d45a78a0635eull, 0x86fb897116c87c35ull, 
  0xd45d35e6ae3d4da1ull, 0x8974836059cca10aull, 0x2bd1a438703fc94cull, 
  0x7b6306a34627ddd0ull, 0x1a3bc84c17b1d543ull, 0x20caba5f1d9e4a94ull, 
  0x547eb47b7282ee9dull, 0xe99e619a4f23aa44ull, 0x6405fa00e2ec94d5ull, 
  0xde83bc408dd3dd05ull, 0x9624ab50b148d446ull, 0x3badd624dd9b0958ull, 
  0xe54ca5d70a80e5d7ull, 0x5e9fcf4ccd211f4dull, 0x7647c32000696720ull, 
  0x29ecd9f40041e074ull, 0xf468107100525891ull, 0x7182148d4066eeb5ull, 
  0xc6f14cd848405531ull, 0xb8ada00e5a506a7dull, 0xa6d90811f0e4851dull, 
  0x908f4a166d1da664ull, 0x9a598e4e043287ffull, 0x40eff1e1853f29feull, 
  0xd12bee59e68ef47dull, 0x82bb74f8301958cfull, 0xe36a52363c1faf02ull, 
  0xdc44e6c3cb279ac2ull, 0x29ab103a5ef8c0baull, 0x7415d448f6b6f0e8ull, 
  0x111b495b3464ad22ull, 0xcab10dd900beec35ull, 0x3d5d514f40eea743ull, 
  0x0cb4a5a3112a5113ull, 0x47f0e785eaba72acull, 0x59ed216765690f57ull, 
  0x306869c13ec3532dull, 0x1e414218c73a13fcull, 0xe5d1929ef90898fbull, 
  0xdf45f746b74abf3aull, 0x6b8bba8c328eb784ull, 0x066ea92f3f326565ull, 
  0xc80a537b0efefebeull, 0xbd06742ce95f5f37ull, 0x2c48113823b73705ull, 
  0xf75a15862ca504c6ull, 0x9a984d73dbe722fcull, 0xc13e60d0d2e0ebbbull, 
  0x318df905079926a9ull, 0xfdf17746497f7053ull, 0xfeb6ea8bedefa634ull, 
  0xfe64a52ee96b8fc1ull, 0x3dfdce7aa3c673b1ull, 0x06bea10ca65c084full, 
  0x486e494fcff30a63ull, 0x5a89dba3c3efccfbull, 0xf89629465a75e01dull, 
  0xf6bbb397f1135824ull, 0x746aa07ded582e2dull, 0xa8c2a44eb4571cddull, 
  0x92f34d62616ce414ull, 0x77b020baf9c81d18ull, 0x0ace1474dc1d122full, 
  0x0d819992132456bbull, 0x10e1fff697ed6c6aull, 0xca8d3ffa1ef463c2ull, 
  0xbd308ff8a6b17cb3ull, 0xac7cb3f6d05ddbdfull, 0x6bcdf07a423aa96cull, 
  0x86c16c98d2c953c7ull, 0xe871c7bf077ba8b8ull, 0x11471cd764ad4973ull, 
  0xd598e40d3dd89bd0ull, 0x4aff1d108d4ec2c4ull, 0xcedf722a585139bbull, 
  0xc2974eb4ee658829ull, 0x733d226229feea33ull, 0x0806357d5a3f5260ull, 
  0xca07c2dcb0cf26f8ull, 0xfc89b393dd02f0b6ull, 0xbbac2078d443ace3ull, 
  0xd54b944b84aa4c0eull, 0x0a9e795e65d4df12ull, 0x4d4617b5ff4a16d6ull, 
  0x504bced1bf8e4e46ull, 0xe45ec2862f71e1d7ull, 0x5d767327bb4e5a4dull, 
  0x3a6a07f8d510f870ull, 0x890489f70a55368cull, 0x2b45ac74ccea842full, 
  0x3b0b8bc90012929eull, 0x09ce6ebb40173745ull, 0xcc420a6a101d0516ull, 
  0x9fa946824a12232eull, 0x47939822dc96abfaull, 0x59787e2b93bc56f8ull, 
  0x57eb4edb3c55b65bull, 0xede622920b6b23f2ull, 0xe95fab368e45eceeull, 
  0x11dbcb0218ebb415ull, 0xd652bdc29f26a11aull, 0x4be76d3346f04960ull, 
  0x6f70a4400c562ddcull, 0xcb4ccd500f6bb953ull, 0x7e2000a41346a7a8ull, 
  0x8ed400668c0c28c9ull, 0x728900802f0f32fbull, 0x4f2b40a03ad2ffbaull, 
  0xe2f610c84987bfa9ull, 0x0dd9ca7d2df4d7caull, 0x91503d1c79720dbcull, 
  0x75a44c6397ce912bull, 0xc986afbe3ee11abbull, 0xfbe85badce996169ull, 
  0xfae27299423fb9c4ull, 0xdccd879fc967d41bull, 0x5400e987bbc1c921ull, 
  0x290123e9aab23b69ull, 0xf9a0b6720aaf6522ull, 0xf808e40e8d5b3e6aull, 
  0xb60b1d1230b20e05ull, 0xb1c6f22b5e6f48c3ull, 0x1e38aeb6360b1af4ull, 
  0x25c6da63c38de1b1ull, 0x579c487e5a38ad0full, 0x2d835a9df0c6d852ull, 
  0xf8e431456cf88e66ull, 0x1b8e9ecb641b5900ull, 0xe272467e3d222f40ull, 
  0x5b0ed81dcc6abb10ull, 0x98e947129fc2b4eaull, 0x3f2398d747b36225ull, 
  0x8eec7f0d19a03aaeull, 0x1953cf68300424adull, 0x5fa8c3423c052dd8ull, 
  0x3792f412cb06794eull, 0xe2bbd88bbee40bd1ull, 0x5b6aceaeae9d0ec5ull, 
  0xf245825a5a445276ull, 0xeed6e2f0f0d56713ull, 0x55464dd69685606cull, 
  0xaa97e14c3c26b887ull, 0xd53dd99f4b3066a9ull, 0xe546a8038efe402aull, 
  0xde98520472bdd034ull, 0x963e66858f6d4441ull, 0xdde7001379a44aa9ull, 
  0x5560c018580d5d53ull, 0xaab8f01e6e10b4a7ull, 0xcab3961304ca70e9ull, 
  0x3d607b97c5fd0d23ull, 0x8cb89a7db77c506bull, 0x77f3608e92adb243ull, 
  0x55f038b237591ed4ull, 0x6b6c46dec52f6689ull, 0x2323ac4b3b3da016ull, 
  0xabec975e0a0d081bull, 0x96e7bd358c904a22ull, 0x7e50d64177da2e55ull, 
  0xdde50bd1d5d0b9eaull, 0x955e4ec64b44e865ull, 0xbd5af13bef0b113full, 
  0xecb1ad8aeacdd58full, 0x67de18eda5814af3ull, 0x80eacf948770ced8ull, 
  0xa1258379a94d028eull, 0x096ee45813a04331ull, 0x8bca9d6e188853fdull, 
  0x775ea264cf55347eull, 0x95364afe032a819eull, 0x3a83ddbd83f52205ull, 
  0xc4926a9672793543ull, 0x75b7053c0f178294ull, 0x5324c68b12dd6339ull, 
  0xd3f6fc16ebca5e04ull, 0x88f4bb1ca6bcf585ull, 0x2b31e9e3d06c32e6ull, 
  0x3aff322e62439fd0ull, 0x09befeb9fad487c3ull, 0x4c2ebe687989a9b4ull, 
  0x0f9d37014bf60a11ull, 0x538484c19ef38c95ull, 0x2865a5f206b06fbaull, 
  0xf93f87b7442e45d4ull, 0xf78f69a51539d749ull, 0xb573440e5a884d1cull, 
  0x31680a88f8953031ull, 0xfdc20d2b36ba7c3eull, 0x3d32907604691b4dull, 
  0xa63f9a49c2c1b110ull, 0x0fcf80dc33721d54ull, 0xd3c36113404ea4a9ull, 
  0x645a1cac083126eaull, 0x3d70a3d70a3d70a4ull, 0xcccccccccccccccdull, 
  0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 
  0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 
  0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 
  0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 
  0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 
  0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 
  0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 
  0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 
  0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 
  0x0000000000000000ull, 0x4000000000000000ull, 0x5000000000000000ull, 
  0xa400000000000000ull, 0x4d00000000000000ull, 0xf020000000000000ull, 
  0x6c28000000000000ull, 0xc732000000000000ull, 0x3c7f400000000000ull, 
  0x4b9f100000000000ull, 0x1e86d40000000000ull, 0x1314448000000000ull, 
  0x17d955a000000000ull, 0x5dcfab0800000000ull, 0x5aa1cae500000000ull, 
  0xf14a3d9e40000000ull, 0x6d9ccd05d0000000ull, 0xe4820023a2000000ull, 
  0xdda2802c8a800000ull, 0xd50b2037ad200000ull, 0x4526f422cc340000ull, 
  0x9670b12b7f410000ull, 0x3c0cdd765f114000ull, 0xa5880a69fb6ac800ull, 
  0x8eea0d047a457a00ull, 0x72a4904598d6d880ull, 0x47a6da2b7f864750ull, 
  0x999090b65f67d924ull, 0xfff4b4e3f741cf6dull, 0xbff8f10e7a8921a5ull, 
  0xaff72d52192b6a0eull, 0x9bf4f8a69f764491ull, 0x02f236d04753d5b5ull, 
  0x01d762422c946591ull, 0x424d3ad2b7b97ef6ull, 0xd2e0898765a7deb3ull, 
  0x63cc55f49f88eb30ull, 0x3cbf6b71c76b25fcull, 0x8bef464e3945ef7bull, 
  0x97758bf0e3cbb5adull, 0x3d52eeed1cbea318ull, 0x4ca7aaa863ee4bdeull, 
  0x8fe8caa93e74ef6bull, 0xb3e2fd538e122b45ull, 0x60dbbca87196b617ull, 
  0xbc8955e946fe31ceull, 0x6babab6398bdbe42ull, 0xc696963c7eed2dd2ull, 
  0xfc1e1de5cf543ca3ull, 0x3b25a55f43294bccull, 0x49ef0eb713f39ebfull, 
  0x6e3569326c784338ull, 0x49c2c37f07965405ull, 0xdc33745ec97be907ull, 
  0x69a028bb3ded71a4ull, 0xc40832ea0d68ce0dull, 0xf50a3fa490c30191ull, 
  0x792667c6da79e0fbull, 0x577001b891185939ull, 0xed4c0226b55e6f87ull, 
  0x544f8158315b05b5ull, 0x696361ae3db1c722ull, 0x03bc3a19cd1e38eaull, 
  0x04ab48a04065c724ull, 0x62eb0d64283f9c77ull, 0x3ba5d0bd324f8395ull, 
  0xca8f44ec7ee3647aull, 0x7e998b13cf4e1eccull, 0x9e3fedd8c321a67full, 
  0xc5cfe94ef3ea101full, 0xbba1f1d158724a13ull, 0x2a8a6e45ae8edc98ull, 
  0xf52d09d71a3293beull, 0x593c2626705f9c57ull, 0x6f8b2fb00c77836dull, 
  0x0b6dfb9c0f956448ull, 0x4724bd4189bd5eadull, 0x58edec91ec2cb658ull, 
  0x2f2967b66737e3eeull, 0xbd79e0d20082ee75ull, 0xecd8590680a3aa12ull, 
  0xe80e6f4820cc9496ull, 0x3109058d147fdcdeull, 0xbd4b46f0599fd416ull, 
  0x6c9e18ac7007c91bull, 0x03e2cf6bc604ddb1ull, 0x84db8346b786151dull, 
  0xe612641865679a64ull, 0x4fcb7e8f3f60c07full, 0xe3be5e330f38f09eull, 
  0x5cadf5bfd3072cc6ull, 0x73d9732fc7c8f7f7ull, 0x2867e7fddcdd9afbull, 
  0xb281e1fd541501b9ull, 0x1f225a7ca91a4227ull, 0x3375788de9b06959ull, 
  0x0052d6b1641c83afull, 0xc0678c5dbd23a49bull, 0xf840b7ba963646e1ull, 
  0xb650e5a93bc3d899ull, 0xa3e51f138ab4cebfull, 0xc66f336c36b10138ull, 
  0xb80b0047445d4185ull, 0xa60dc059157491e6ull, 0x87c89837ad68db30ull, 
  0x29babe4598c311fcull, 0xf4296dd6fef3d67bull, 0x1899e4a65f58660dull, 
  0x5ec05dcff72e7f90ull, 0x76707543f4fa1f74ull, 0x6a06494a791c53a9ull, 
  0x0487db9d17636893ull, 0x45a9d2845d3c42b7ull, 0x0b8a2392ba45a9b3ull, 
  0x8e6cac7768d7141full, 0x3207d795430cd927ull, 0x7f44e6bd49e807b9ull, 
  0x5f16206c9c6209a7ull, 0x36dba887c37a8c10ull, 0xc2494954da2c978aull, 
  0xf2db9baa10b7bd6dull, 0x6f92829494e5acc8ull, 0xcb772339ba1f17faull, 
  0xff2a760414536efcull, 0xfef5138519684abbull, 0x7eb258665fc25d6aull, 
  0xef2f773ffbd97a62ull, 0xaafb550ffacfd8fbull, 0x95ba2a53f983cf39ull, 
  0xdd945a747bf26184ull, 0x94f971119aeef9e5ull, 0x7a37cd5601aab85eull, 
  0xac62e055c10ab33bull, 0x577b986b314d600aull, 0xed5a7e85fda0b80cull, 
  0x14588f13be847308ull, 0x596eb2d8ae258fc9ull, 0x6fca5f8ed9aef3bcull, 
  0x25de7bb9480d5855ull, 0xaf561aa79a10ae6bull, 0x1b2ba1518094da05ull, 
  0x90fb44d2f05d0843ull, 0x353a1607ac744a54ull, 0x42889b8997915ce9ull, 
  0x69956135febada12ull, 0x43fab9837e699096ull, 0x94f967e45e03f4bcull, 
  0x1d1be0eebac278f6ull, 0x6462d92a69731733ull, 0x7d7b8f7503cfdcffull, 
  0x5cda735244c3d43full, 0x3a0888136afa64a8ull, 0x088aaa1845b8fdd1ull, 
  0x8aad549e57273d46ull, 0x36ac54e2f678864cull, 0x84576a1bb416a7deull, 
  0x656d44a2a11c51d6ull, 0x9f644ae5a4b1b326ull, 0x873d5d9f0dde1fefull, 
  0xa90cb506d155a7ebull, 0x09a7f12442d588f3ull, 0x0c11ed6d538aeb30ull, 
  0x8f1668c8a86da5fbull, 0xf96e017d694487bdull, 0x37c981dcc395a9adull, 
  0x85bbe253f47b1418ull, 0x93956d7478ccec8full, 0x387ac8d1970027b3ull, 
  0x06997b05fcc0319full, 0x441fece3bdf81f04ull, 0xd527e81cad7626c4ull, 
  0x8a71e223d8d3b075ull, 0xf6872d5667844e4aull, 0xb428f8ac016561dcull, 
  0xe13336d701beba53ull, 0xecc0024661173474ull, 0x27f002d7f95d0191ull, 
  0x31ec038df7b441f5ull, 0x7e67047175a15272ull, 0x0f0062c6e984d387ull, 
  0x52c07b78a3e60869ull, 0xa7709a56ccdf8a83ull, 0x88a66076400bb692ull, 
  0x6acff893d00ea436ull, 0x0583f6b8c4124d44ull, 0xc3727a337a8b704bull, 
  0x744f18c0592e4c5dull, 0x1162def06f79df74ull, 0x8addcb5645ac2ba9ull, 
  0x6d953e2bd7173693ull, 0xc8fa8db6ccdd0438ull, 0x1d9c9892400a22a3ull, 
  0x2503beb6d00cab4cull, 0x2e44ae64840fd61eull, 0x5ceaecfed289e5d3ull, 
  0x7425a83e872c5f48ull, 0xd12f124e28f7771aull, 0x82bd6b70d99aaa70ull, 
  0x636cc64d1001550cull, 0x3c47f7e05401aa4full, 0x65acfaec34810a72ull, 
  0x7f1839a741a14d0eull, 0x1ede48111209a051ull, 0x934aed0aab460433ull, 
  0xf81da84d56178540ull, 0x36251260ab9d668full, 0xc1d72b7c6b42601aull, 
  0xb24cf65b8612f820ull, 0xdee033f26797b628ull, 0x169840ef017da3b2ull, 
  0x8e1f289560ee864full, 0xf1a6f2bab92a27e3ull, 0xae10af696774b1dcull, 
  0xacca6da1e0a8ef2aull, 0x17fd090a58d32af4ull, 0xddfc4b4cef07f5b1ull, 
  0x4abdaf101564f98full, 0x9d6d1ad41abe37f2ull, 0x84c86189216dc5eeull, 
  0x32fd3cf5b4e49bb5ull, 0x3fbc8c33221dc2a2ull, 0x0fabaf3feaa5334bull, 
  0x29cb4d87f2a7400full, 0x743e20e9ef511013ull, 0x914da9246b255417ull, 
  0x1ad089b6c2f7548full, 0xa184ac2473b529b2ull, 0xc9e5d72d90a2741full, 
  0x7e2fa67c7a658893ull, 0xddbb901b98feeab8ull, 0x552a74227f3ea566ull, 
  0xd53a88958f872760ull, 0x8a892abaf368f138ull, 0x2d2b7569b0432d86ull, 
  0x9c3b29620e29fc74ull, 0x8349f3ba91b47b90ull, 0x241c70a936219a74ull, 
  0xed238cd383aa0111ull, 0xf4363804324a40abull, 0xb143c6053edcd0d6ull, 
  0xdd94b7868e94050bull, 0xca7cf2b4191c8327ull, 0xfd1c2f611f63a3f1ull, 
  0xbc633b39673c8cedull, 0xd5be0503e085d814ull, 0x4b2d8644d8a74e19ull, 
  0xddf8e7d60ed1219full, 0xcabb90e5c942b504ull, 0x3d6a751f3b936244ull, 
  0x0cc512670a783ad5ull, 0x27fb2b80668b24c6ull, 0xb1f9f660802dedf7ull, 
  0x5e7873f8a0396974ull, 0xdb0b487b6423e1e9ull, 0x91ce1a9a3d2cda63ull, 
  0x7641a140cc7810fcull, 0xa9e904c87fcb0a9eull, 0x546345fa9fbdcd45ull, 
  0xa97c177947ad4096ull, 0x49ed8eabcccc485eull, 0x5c68f256bfff5a75ull, 
  0x73832eec6fff3112ull, 0xc831fd53c5ff7eacull, 0xba3e7ca8b77f5e56ull, 
  0x28ce1bd2e55f35ecull, 0x7980d163cf5b81b4ull, 0xd7e105bcc3326220ull, 
  0x8dd9472bf3fefaa8ull, 0xb14f98f6f0feb952ull, 0x6ed1bf9a569f33d4ull, 
  0x0a862f80ec4700c9ull, 0xcd27bb612758c0fbull, 0x8038d51cb897789dull, 
  0xe0470a63e6bd56c4ull, 0x1858ccfce06cac75ull, 0x0f37801e0c43ebc9ull, 
  0xd30560258f54e6bbull, 0x47c6b82ef32a206aull, 0x4cdc331d57fa5442ull, 
  0xe0133fe4adf8e953ull, 0x58180fddd97723a7ull, 0x570f09eaa7ea7649ull, 
  0x2cd2cc6551e513dbull, 0xf8077f7ea65e58d2ull, 0xfb04afaf27faf783ull, 
  0x79c5db9af1f9b564ull, 0x18375281ae7822bdull, 0x8f2293910d0b15b6ull, 
  0xb2eb3875504ddb23ull, 0x5fa60692a46151ecull, 0xdbc7c41ba6bcd334ull, 
  0x12b9b522906c0801ull, 0xd768226b34870a01ull, 0xe6a1158300d46641ull, 
  0x60495ae3c1097fd1ull, 0x385bb19cb14bdfc5ull, 0x46729e03dd9ed7b6ull, 
  0x6c07a2c26a8346d2ull, 0xc7098b7305241886ull, 0xb8cbee4fc66d1ea8ull, 
  0x737f74f1dc043329ull, 0x505f522e53053ff3ull, 0x647726b9e7c68ff0ull, 
  0x5eca783430dc19f6ull, 0xb67d16413d132073ull, 0xe41c5bd18c57e890ull, 
  0x8e91b962f7b6f15aull, 0x723627bbb5a4adb1ull, 0xcec3b1aaa30dd91dull, 
  0x213a4f0aa5e8a7b2ull, 0xa988e2cd4f62d19eull, 0x93eb1b80a33b8606ull, 
  0xbc72f130660533c4ull, 0xeb8fad7c7f8680b5ull, 0xa67398db9f6820e2ull, 
  0x88083f8943a1148dull, 0x6a0a4f6b948959b1ull, 0x848ce34679abb01dull, 
  0xf2d80e0c0c0b4e12ull, 0x6f8e118f0f0e2196ull, 0x4b7195f2d2d1a9fcull, 
  0x8f26fdb7c3c30a3eull, 
};

static const char dt_pairs[200] =
    "00010203040506070809101112131415161718192021222324"
    "25262728293031323334353637383940414243444546474849"
    "50515253545556575859606162636465666768697071727374"
    "75767778798081828384858687888990919293949596979899";

// dt_uint writes v backwards into tmp[0..20) and returns the start index.
static int dt_uint(u8 *tmp, u64 v) {
  int i = 20;
  while (v >= 100) {
    u64 q = v / 100;
    u64 r = (v - q * 100) * 2;
    i -= 2;
    tmp[i] = (u8)dt_pairs[r];
    tmp[i + 1] = (u8)dt_pairs[r + 1];
    v = q;
  }
  if (v >= 10) {
    u64 r = v * 2;
    i -= 2;
    tmp[i] = (u8)dt_pairs[r];
    tmp[i + 1] = (u8)dt_pairs[r + 1];
  } else {
    i--;
    tmp[i] = (u8)('0' + v);
  }
  return i;
}

static inline u64 dt_round_odd(u64 ghi, u64 glo, u64 cp) {
  u128 x = (u128)cp * glo;
  u128 y = (u128)cp * ghi + (u64)(x >> 64);
  u64 ylo = (u64)y;
  u64 yhi = (u64)(y >> 64);
  return yhi | (u64)(ylo > 1);
}

// dt_schubfach: sig * 10^exp, shortest round-tripping decimal for the
// finite non-zero float64 whose raw bits are given.
static void dt_schubfach(u64 bitsIn, u64 *sig, int *exp) {
  const u64 sigMask = ((u64)1 << 52) - 1;
  u64 rsig = bitsIn & sigMask;
  int rexp = (int)((bitsIn >> 52) & 0x7ff);

  u64 c;
  int q;
  if (rexp != 0) {
    c = rsig | ((u64)1 << 52);
    q = rexp - 1075;
  } else {
    c = rsig;
    q = 1 - 1075;
  }

  int even = (c & 1) == 0;
  int irregular = rsig == 0 && rexp > 1;

  u64 cbl = irregular ? 4 * c - 1 : 4 * c - 2;
  u64 cb = 4 * c;
  u64 cbr = 4 * c + 2;

  int k = q * 1262611;
  if (irregular) k -= 524031;
  k >>= 22;

  int h = q + ((-k * 1741647) >> 19) + 1;
  u64 phi = dt_pow10_hi[-k - POW10_MIN];
  u64 plo = dt_pow10_lo[-k - POW10_MIN];
  u64 vbl = dt_round_odd(phi, plo, cbl << h);
  u64 vb = dt_round_odd(phi, plo, cb << h);
  u64 vbr = dt_round_odd(phi, plo, cbr << h);

  u64 lower = vbl, upper = vbr;
  if (!even) {
    lower++;
    upper--;
  }

  u64 s = vb / 4;
  if (s >= 10) {
    u64 sp = s / 10;
    int upInside = lower <= 40 * sp;
    int wpInside = 40 * sp + 40 <= upper;
    if (upInside != wpInside) {
      *sig = sp + (u64)wpInside;
      *exp = k + 1;
      return;
    }
  }

  int uInside = lower <= 4 * s;
  int wInside = 4 * s + 4 <= upper;
  if (uInside != wInside) {
    *sig = s + (u64)wInside;
    *exp = k;
    return;
  }

  u64 mid = 4 * s + 2;
  int roundUp = vb > mid || (vb == mid && (s & 1) != 0);
  *sig = s + (u64)roundUp;
  *exp = k;
}

// simd_dtoa_f64 renders v exactly as simdjson's appendFloat does for a
// finite float64, and returns the byte count in *out. dst needs 25 bytes.
void simd_dtoa_f64(i64 *__restrict out, u8 *__restrict dst, double v) {
  u64 bits;
  __builtin_memcpy(&bits, &v, 8);
  int neg = (int)(bits >> 63);
  u64 abits = bits & ~((u64)1 << 63);
  f64 abs;
  __builtin_memcpy(&abs, &abits, 8);
  isize n = 0;

  // Zero, either sign.
  if (abits == 0) {
    if (neg) dst[n++] = '-';
    dst[n++] = '0';
    *out = n;
    return;
  }

  // Whole numbers below 1e15 print as integers: the conversion round trip
  // is exact there, so the integer is also the shortest decimal.
  if (abs < 1e15) {
    i64 t = (i64)abs;
    if ((f64)t == abs) {
      if (neg) dst[n++] = '-';
      u8 tmp[20];
      int i = dt_uint(tmp, (u64)t);
      for (; i < 20; i++) dst[n++] = tmp[i];
      *out = n;
      return;
    }
  }

  u64 sig;
  int exp;
  dt_schubfach(abits, &sig, &exp);

  // Digits with trailing zeros stripped.
  u8 tmp[20];
  int start = dt_uint(tmp, sig);
  int len = 20 - start;
  while (len > 1 && tmp[start + len - 1] == '0') {
    len--;
    exp++;
  }
  if (neg) dst[n++] = '-';
  int point = exp + len;

  int expFormat = abs < 1e-6 || abs >= 1e21;
  if (expFormat) {
    dst[n++] = tmp[start];
    if (len > 1) {
      dst[n++] = '.';
      for (int i = 1; i < len; i++) dst[n++] = tmp[start + i];
    }
    dst[n++] = 'e';
    int e = point - 1;
    if (e < 0) {
      dst[n++] = '-';
      e = -e;
    } else {
      dst[n++] = '+';
    }
    if (e < 10) {
      dst[n++] = (u8)('0' + e);
    } else {
      u8 et[20];
      int i = dt_uint(et, (u64)e);
      for (; i < 20; i++) dst[n++] = et[i];
    }
    *out = n;
    return;
  }

  if (point <= 0) {
    dst[n++] = '0';
    dst[n++] = '.';
    for (int i = 0; i < -point; i++) dst[n++] = '0';
    for (int i = 0; i < len; i++) dst[n++] = tmp[start + i];
  } else if (point >= len) {
    for (int i = 0; i < len; i++) dst[n++] = tmp[start + i];
    for (int i = len; i < point; i++) dst[n++] = '0';
  } else {
    for (int i = 0; i < point; i++) dst[n++] = tmp[start + i];
    dst[n++] = '.';
    for (int i = 0; i < len - point; i++) dst[n++] = tmp[start + point + i];
  }
  *out = n;
}
