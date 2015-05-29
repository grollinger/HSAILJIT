__kernel void sum(int size, __global double* a, __global double* b) {
	int i = get_global_id(0);
    a[i] += b[i];
}
