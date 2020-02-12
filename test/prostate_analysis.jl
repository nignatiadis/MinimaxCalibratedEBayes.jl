using MinimaxCalibratedEBayes
using RCall
using Plots
pgfplotsx()


prostz_file = MinimaxCalibratedEBayes.prostate_data_file

R"load($prostz_file)"
@rget prostz;
Zs = StandardNormalSample.(prostz)
