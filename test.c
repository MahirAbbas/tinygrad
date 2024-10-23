void E_(__fp16* restrict data0, __fp16* restrict data1, __fp16* restrict data2) {
  __fp16 val0 = *(data1+0);
  __fp16 val1 = *(data2+0);
  *(data0+0) = (__fp16)((float)((val0*val1)));
}
