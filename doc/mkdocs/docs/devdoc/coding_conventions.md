# Coding Conventions

The following coding conventions are followed when developing Shamrock. In practice, there may be slight deviations from these guidelines 😅. Please notify or raise an issue if these conventions are not followed somewhere in the code.

## C++ Style Guide

### General Rules
- No tabs (use spaces for indentation)
- No raw pointers without wrapper or smart pointer
- Use `T{}` for zero initialization of template types instead of `T(0)` to ensure compatibility with vectors and other complex types

## Naming Conventions

### Primitive Types

Primitive types are basic types representable by the actual hardware, typically integers, floats, and SYCL vectors.

Since Shamrock uses binary manipulation extensively, all types are named with a prefix (`u` for unsigned, `i` for signed integers, `f` for floats) followed by the number of bits. This can optionally be followed by `_x` where `x` is the number of elements in a vector.

**Primitive types:** `i64`, `i32`, `i16`, `i8`, `u64`, `u32`, `u16`, `u8`, `f16`, `f32`, `f64`

**Vector examples:** `f64_3`, `u64_16`, ...

### Classes, Structs, and Enums

Classes, structs, and enums in Shamrock follow PascalCase naming scheme, where each word starts with a capital letter.

**Example:** `IMeanIKindaLikeThisCaseTheOthersAreLessReadableToMe`

### Functions

Functions in Shamrock use snake_case to distinguish them from classes.

**Example:** `is_this_informatics_or_physics(...)`

## Template Conventions

### Vector and Scalar Templates

Since many models can be implemented in Shamrock, utilities/classes are implemented for any primitive types. Generic classes use the following pattern:

```c++
template<class Tvec>
class Whateva {
    using Tscal = shambase::VecComponent<Tvec>;
    static constexpr u32 dimension = shambase::VectorProperties<Tvec>::dimension;
}
```

`Tvec` is sufficient to infer both the scalar type and the dimension, simplifying the template.

**Conventions:**
- `Tscal` for template scalar types
- `Tvec` for template vector types

### Special Template Types

#### Morton & Hilbert Codes

Morton codes and Hilbert codes shall be named `Tmorton` and `THilbert` respectively, since they will be templated.
