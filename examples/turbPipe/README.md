# LES of a round turbulent pipe flow at Re_tau=550
# using velocity recycling + stabilized outflow

Reference: 
[1] ftp://ftp.mech.kth.se/pub/pschlatt/DATA/PIPE/ 

The supplied parameters are matched to `../picard_turbpipe` for a direct
1000-step comparison at fixed `dt=0.001`. Both cases export instantaneous line
profiles every 100 accepted steps through the shared `lineProfiles.hpp`.
The native solver algorithm is unchanged. See
[`../picard_turbpipe/README.md`](../picard_turbpipe/README.md#direct-comparison-with-the-original-turbpipe-case)
for line locations, identical-mesh/run requirements and the overlay/metrics script.
