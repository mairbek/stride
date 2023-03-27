Implementing basics from http://blog.ezyang.com/2019/05/pytorch-internals/

- [x] Move most of the logic to `View`.
- [x] Add toList and toString.
- [x] Implement `__eq__` that works for any dims. 
- [ ] Broadcasting: stride is zero
- [ ] Transpose: strides are swapped
- [ ] Flip: negative strides (storage offset must be adjusted accordingly)
- [ ] Diagonal: stride is one greater than size
- [ ] Rolling window: stride is less than size
