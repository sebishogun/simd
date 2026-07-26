// The code-generation toolchain lives in its own module so that consumers of
// the library never inherit any of it. `go get github.com/sebishogun/simd`
// pulls in golang.org/x/sys and nothing else; clang, the object-file reader
// and the assembly emitter are contributor tooling.
//
// This is the pattern segmentio/asm uses to keep avo out of its users' builds,
// and it is the reason the generated .s files are committed rather than built
// on demand.
module github.com/sebishogun/simd/tools

go 1.26
