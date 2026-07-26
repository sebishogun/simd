// Package astcheck validates that the C source agrees with the manifest.
//
// # Why the manifest is not derived from the AST
//
// The obvious alternative — read the C declarations and generate everything
// from them, as gorse-io/goat does — cannot actually work, because the C
// signature does not contain the information the generator needs.
//
//	void simd_add_f32(float *d, const float *a, const float *b, long n);
//
// Nothing here says whether n is len(d) or len(a), whether the three pointers
// are three Go slices or one slice and two arrays, or which of them is the
// destination. That mapping is a design decision, and it lives in the manifest
// because there is nowhere else it could live.
//
// # Why the AST is still worth reading
//
// The manifest and the C can drift. Add a parameter to a kernel and forget to
// update the declaration, and the generator will happily build a prologue that
// loads arguments into the wrong registers — no compiler error, no assembler
// error, just a kernel reading whatever was in the register the ABI assigned
// to some other value. That is the sort of bug that survives review and shows
// up as corrupted numbers weeks later.
//
// So the AST is used to check, not to generate. Only top-level function
// declarations are read, and only their name, parameter list and return type.
// That is the stable, boring part of clang's output; goat's fragility — the
// `static inline` helpers, single-line if statements, union type-punning and
// array initializers its README warns about — all comes from trying to walk
// function bodies, which nothing here does.
package astcheck

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"

	"github.com/sebishogun/simd/tools/simdgen/spec"
)

// Signature is one C function declaration as clang reports it.
type Signature struct {
	Name string
	// Result is the return type, "void" for a procedure.
	Result string
	// Params are the parameter types in declaration order, with qualifiers
	// stripped: "float *", "const float *" and "float * restrict" all reduce
	// to "float*".
	Params []string
}

// IsPointer reports whether parameter i is a pointer.
func (s Signature) IsPointer(i int) bool {
	return i < len(s.Params) && strings.HasSuffix(s.Params[i], "*")
}

// IsFloatScalar reports whether parameter i is a float or double passed by
// value, which is what decides between the integer and floating-point argument
// register sequences.
func (s Signature) IsFloatScalar(i int) bool {
	if i >= len(s.Params) || s.IsPointer(i) {
		return false
	}
	return s.Params[i] == "float" || s.Params[i] == "double"
}

// clang's JSON AST. Only the handful of fields needed are declared; everything
// else in the node is ignored, which is what keeps this insensitive to clang
// version changes.
type astNode struct {
	Kind        string    `json:"kind"`
	Name        string    `json:"name"`
	StorageMode string    `json:"storageClass"`
	Type        astType   `json:"type"`
	Inner       []astNode `json:"inner"`
}

type astType struct {
	QualType string `json:"qualType"`
}

// Parse reads the top-level function declarations from a C file.
func Parse(clangPath, srcPath string, flags []string) (map[string]Signature, error) {
	args := append([]string{}, flags...)
	args = append(args, "-Xclang", "-ast-dump=json", "-fsyntax-only", srcPath)

	cmd := exec.Command(clangPath, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("astcheck: %s: %w\n%s", srcPath, err, stderr.String())
	}

	var tu astNode
	if err := json.Unmarshal(stdout.Bytes(), &tu); err != nil {
		return nil, fmt.Errorf("astcheck: parsing clang's AST for %s: %w", srcPath, err)
	}

	out := map[string]Signature{}
	for _, n := range tu.Inner {
		if n.Kind != "FunctionDecl" || n.Name == "" {
			continue
		}
		// Skip declarations without a body: a prototype is not something to
		// generate from.
		hasBody := false
		for _, in := range n.Inner {
			if in.Kind == "CompoundStmt" {
				hasBody = true
				break
			}
		}
		if !hasBody {
			continue
		}
		sig, ok := parseQualType(n.Name, n.Type.QualType)
		if !ok {
			continue
		}
		out[n.Name] = sig
	}
	return out, nil
}

// parseQualType splits clang's rendering of a function type, which is of the
// form "void (float *, const float *, long)".
func parseQualType(name, qual string) (Signature, bool) {
	open := strings.Index(qual, "(")
	close := strings.LastIndex(qual, ")")
	if open < 0 || close < open {
		return Signature{}, false
	}
	sig := Signature{
		Name:   name,
		Result: normalize(qual[:open]),
	}
	inner := strings.TrimSpace(qual[open+1 : close])
	if inner == "" || inner == "void" {
		return sig, true
	}
	for _, p := range strings.Split(inner, ",") {
		sig.Params = append(sig.Params, normalize(p))
	}
	return sig, true
}

// normalize strips qualifiers and whitespace so that "const float *restrict"
// and "float*" compare equal. Only the shape matters here — pointer versus
// scalar, and float versus integer — not constness.
func normalize(t string) string {
	t = strings.TrimSpace(t)
	for _, q := range []string{"const ", "volatile ", "restrict ", "__restrict "} {
		t = strings.ReplaceAll(t, q, "")
	}
	t = strings.ReplaceAll(t, "__restrict", "")
	t = strings.ReplaceAll(t, "restrict", "")
	t = strings.Join(strings.Fields(t), "")
	return t
}

// Check reports every disagreement between the manifest and the C source.
//
// It verifies four things, each of which would otherwise fail silently:
//
//   - the function exists at all, so a renamed kernel is caught before the
//     object file is searched for a symbol that is not there;
//   - the number of C arguments matches, so an added or removed parameter
//     cannot shift every later argument into the wrong register;
//   - each argument is a pointer where the manifest passes a pointer and a
//     scalar where it passes a value, which is what decides between the
//     integer and floating-point register sequences;
//   - the return type is void, because kernels write results through a
//     pointer rather than returning them — see spec.ResultAddr.
func Check(kernels []spec.Kernel, sigs map[string]Signature) []error {
	var errs []error
	for _, k := range kernels {
		sig, ok := sigs[k.CName]
		if !ok {
			errs = append(errs, fmt.Errorf("%s: no such function in the C source "+
				"(manifest says %s)", k.CName, k.GoName))
			continue
		}
		if len(sig.Params) != len(k.CArgs) {
			errs = append(errs, fmt.Errorf(
				"%s: C takes %d argument(s) %v but the manifest passes %d — every "+
					"argument after the mismatch would land in the wrong register",
				k.CName, len(sig.Params), sig.Params, len(k.CArgs)))
			continue
		}
		if sig.Result != "void" {
			errs = append(errs, fmt.Errorf(
				"%s: returns %s, but kernels must return void and write through an "+
					"out-pointer. The generator cannot append a store after the body, "+
					"because LLVM lays basic blocks out past the return instruction",
				k.CName, sig.Result))
		}
		for i, ca := range k.CArgs {
			wantPtr := ca.Part == spec.Base || ca.Part == spec.ResultAddr
			if got := sig.IsPointer(i); got != wantPtr {
				what := "a pointer"
				if !wantPtr {
					what = "a scalar"
				}
				errs = append(errs, fmt.Errorf(
					"%s: argument %d is %q in C but the manifest passes %s (%v)",
					k.CName, i, sig.Params[i], what, ca.Part))
				continue
			}
			// A float scalar travels in a floating-point register and an
			// integer scalar in a general one. Getting this backwards reads
			// an unrelated register, which is silent and wrong.
			if ca.Part == spec.Value {
				p, _ := k.Param(ca.From)
				if want := p.Type.IsFloat(); sig.IsFloatScalar(i) != want {
					errs = append(errs, fmt.Errorf(
						"%s: argument %d is %q in C but the manifest passes Go %s — "+
							"one uses the floating-point argument registers and the "+
							"other the integer ones",
						k.CName, i, sig.Params[i], p.Type.GoString()))
				}
			}
		}
	}
	return errs
}
