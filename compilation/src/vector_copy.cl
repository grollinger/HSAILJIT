__kernel void copy(__global unsigned int* a, __global unsigned int* b) {
	int i = get_global_id(0);

    a[i] = b[i];
}
