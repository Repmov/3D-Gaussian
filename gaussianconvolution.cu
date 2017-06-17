/*This code is licensed under the:  Creative Commons - Attribution-Share Alike 2.0 UK: England & Wales  License
  Please see the full license at: http://www.bv2.co.uk/?page_id=849
  or the license certificate at:  http://creativecommons.org/licenses/by-sa/2.0/uk/
  contact Barrett@bv2.co.uk for other permissions
*/


__constant__ float d_G1D[5];


void calcGaussianCoefficients()
{
	//create the 1D kernel co-efficients
	float G1D[5];
	int g1dindex = 0;
	total=0;
	for (int z=-2;z<=2;++z)
	{
		float factor = exp(-((z*z)/(2*1.4*1.4)));   
		G1D[g1dindex] = factor;
		g1dindex++;
		total+=factor;
	}
	cudaMemcpyToSymbol(d_G1D, G1D, sizeof(G1D));
}



__global__
void calcGaussianXYPlaneConvolution256(float* d_volumeChunk,float* d_tempVolumeChunk)
{
  //assume each plane is 256x256
	//  each thread does a full column
	uint plane = blockIdx.x;       //the plane
	uint col = threadIdx.x;         //the column

	__shared__ float yConvol[256];  //256

	float valCache[5];  //should be stored in reg and not lmem - but always good idea to check ptx / cubin output
	valCache[0] = 0;
	valCache[1] = 0;
	valCache[2] = 0;
	valCache[3] = 0;
	valCache[4] = 0;

	for (int c=0;c<256;++c)
	{
		float inputVal = d_volumeChunk[plane*256*256+c*256+col];  
		
		valCache[0]=valCache[1];
		valCache[1]=valCache[2];
		valCache[2]=valCache[3];
		valCache[3]=valCache[4];
		valCache[4]=inputVal;
		
		if (c>1)    //we have enough data to produce a value
		{
			float outputVal=valCache[0]*d_G1D[0]+valCache[1]*d_G1D[1]+valCache[2]*d_G1D[2]+valCache[3]*d_G1D[3]+valCache[4]*d_G1D[4];
			yConvol[col] = outputVal;

			__syncthreads(); //dangerous to put this in a loop/conditional but in this case should be fine - can probably do without it as 1/2 warp will be sync'd

			//now calc the x convol
			float xConvol = 0;
			if ((col>1) && (col<256-1))
			{
			   xConvol  = yConvol[col-2]*d_G1D[0]+yConvol[col-1]*d_G1D[1]+yConvol[col]*d_G1D[2]+yConvol[col+1]*d_G1D[3]+yConvol[col+2]*d_G1D[4];
			}

			//__syncthreads(); - not really needed here but syncs the writes
			d_tempVolumeChunk[plane*256*256+c*256+col] = xConvol;  //coalesced
		}
	}

}
