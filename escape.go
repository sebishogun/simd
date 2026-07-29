package simd

// JSON string escaping.
//
// # The shape: append, and scan-then-copy
//
// Escaping is two very different workloads glued together. Almost every byte
// of real input needs no escaping and wants to be copied at memory speed; the
// rare byte that does need it wants a small table. So the loop is not
// byte-at-a-time — it is [IndexAny] over the 34-byte set that needs escaping
// (quote, backslash, and the 32 control bytes), which is already accelerated,
// with verbatim copies of everything between hits. On clean input this is one
// scan and one copy; encoding/json's per-byte state machine is what it
// replaces.
//
// The API is the append convention of strconv.AppendQuote, because the output
// length is data-dependent and append is how Go spells that without forcing an
// allocation per call: bring a buffer with capacity and it is reused.
//
// # What it escapes, exactly
//
// Quote and backslash as \" and \\; \b \f \n \r \t by name; the remaining
// control bytes as \u00XX. Bytes 0x80 and above are copied verbatim — valid
// UTF-8 stays valid UTF-8, and JSON does not require escaping it. This matches
// encoding/json with SetEscapeHTML(false); the HTML-paranoid escaping of < >
// and & is deliberately not done, because it exists for embedding JSON in
// HTML, doubles the escape set, and callers who need it know they need it.

// escapeJSONSet is the set of bytes that cannot appear raw in a JSON string:
// the quote, the backslash, and every control byte below 0x20.
const escapeJSONSet = "\"\\" +
	"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f" +
	"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f"

const hexDigits = "0123456789abcdef"

// AppendEscapeJSON appends s to dst with JSON string escaping applied — the
// body of a JSON string, without the surrounding quotes — and returns the
// extended slice.
//
//	buf := make([]byte, 0, 4096)          // once
//	for _, rec := range records {
//	    buf = append(buf[:0], '"')
//	    buf = simd.AppendEscapeJSON(buf, rec.Name)
//	    buf = append(buf, '"')
//	    ...
//	}
//
// Escapes are the standard ones: \" \\ \b \f \n \r \t, and \u00XX for the
// other control bytes. Bytes 0x80 and above pass through, so valid UTF-8 in
// gives valid UTF-8 out. It does not escape < > & — see the package note; use
// encoding/json if you are embedding the result in HTML.
func AppendEscapeJSON[S Text](dst []byte, s S) []byte {
	b := textBytes(s)
	for len(b) > 0 {
		i := IndexAny(b, escapeJSONSet)
		if i < 0 {
			return append(dst, b...)
		}
		dst = append(dst, b[:i]...)
		c := b[i]
		switch c {
		case '"':
			dst = append(dst, '\\', '"')
		case '\\':
			dst = append(dst, '\\', '\\')
		case '\b':
			dst = append(dst, '\\', 'b')
		case '\f':
			dst = append(dst, '\\', 'f')
		case '\n':
			dst = append(dst, '\\', 'n')
		case '\r':
			dst = append(dst, '\\', 'r')
		case '\t':
			dst = append(dst, '\\', 't')
		default:
			dst = append(dst, '\\', 'u', '0', '0',
				hexDigits[c>>4], hexDigits[c&0xf])
		}
		b = b[i+1:]
	}
	return dst
}

// NeedsEscapeJSON reports whether s contains any byte that AppendEscapeJSON
// would change. The common answer on real data is no, and knowing it costs one
// accelerated scan — a serializer can then write the input directly.
func NeedsEscapeJSON[S Text](s S) bool {
	return IndexAny(textBytes(s), escapeJSONSet) >= 0
}
