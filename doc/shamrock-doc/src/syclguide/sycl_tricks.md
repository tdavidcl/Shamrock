# Sycl tricks

## Dealing with sycl accessors in loop macros

### MPI



### SYCL parralel for

```cpp
shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){
                
    sycl::accessor ...

    passing variables

    cgh.parallel_for([](sycl::range) ...,[=](sycl::item ...){
        ... kernel space
    });
});
```

### Accessor class design

```cpp

struct CustomAcc{
    sycl::accessor ...
    passing variables
};

//withing a function templated on the accessor and the functor
shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){
                
    CustomAcc local = acc_arg;

    cgh.parallel_for([](sycl::range) ...,[=](sycl::item ... gid){
       func(gid, acc_arg);
    });
});
```