//-------------------------------------------
// Function to generate mass grid 
//-------------------------------------------
#include "options.hpp"


void init_grid(flt massmin, flt massmax,u32 nbins,flt* massgrid,flt* massbins, flt* xmeanlog){
    flt r = pow(massmax/massmin,((flt)1)/((flt)nbins));
    massgrid[0] = massmin;
    for (unsigned int j=0;j<nbins;j++){
        massgrid[j+1] = r*massgrid[j];
    }
    for (unsigned int j=0;j<nbins;j++){
        massbins[j] = (massgrid[j+1]+massgrid[j])/((flt)2);
        xmeanlog[j] = sqrt(massgrid[j+1]*massgrid[j]);
    }
}