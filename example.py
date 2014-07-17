import pyRAVE
import pylab as plb
import numpy as np
import healpy

# Load RAVE data
RAVE = pyRAVE.readCSV('RAVE_DR4.csv')

# =======================================================
# Selection criteria by the user

user_select = np.ones(len(RAVE['rave_obs_id']),dtype=bool) # all entries
user_select = (RAVE['snr_k']>20) &\
              (RAVE['ehrv'] < 8) &\
              (RAVE['algo_conv_k'] == 0)



# =======================================================

# Compute approximated I magnitude
RAVE['I2MASS'] = pyRAVE.computeI2MASS(RAVE['jmag_2mass'],RAVE['kmag_2mass'])
# Weed out duplicates
unique = pyRAVE.findUnique(RAVE['raveid'],RAVE['rave_obs_id'],RAVE['snr_k'])
print np.sum(unique==False), " entries lost from repeat observations."
# Weed out targets that are inconsistent with the color cut
colour_cut = pyRAVE.JmK_color_cut(RAVE['b'],
                                  RAVE['jmag_2mass'],
                                  RAVE['kmag_2mass'])
print np.sum(unique&(colour_cut==False)), " entries lost from colour cut."

# Remove sky regions that were not systematically observed
unproblematic = pyRAVE.remove_problematic_fields(RAVE['l'],RAVE['b'])
print np.sum(unique&colour_cut&(unproblematic==False)), \
    " entries lost from invalid sky region removal."


print "-------------------"
print np.sum(unique & colour_cut & unproblematic & (user_select==False)),\
    " stars lost from user defined selection."
use = unique & colour_cut & unproblematic & user_select
print "-------------------"
print np.sum(use), " stars left."

# Compute HEALPIX indices
RAVE['healpix_ids'] = pyRAVE.computeHEALPIX_ids(RAVE['radeg_2mass'],
                                                RAVE['dedeg_2mass'])

# Evaluate completeness in (ra,dec,I2mass) bins
comp,Irange = pyRAVE.computeCompleteness(RAVE['healpix_ids'][use],
                                         RAVE['I2MASS'][use],
                                         RAVE['jmag_2mass'][use],
                                         RAVE['kmag_2mass'][use],
                                         0.2)


# Write into ASCII file
ofname = 'RAVE_completeness.txt'
f = open(ofname,'w')

for I in Irange:
    f.write("%.2f "%I)
f.write("\n")

for i in range(len(comp)):
    for j in range(np.shape(comp)[1]):
        f.write("%.5f "%(comp[i][j]))
    f.write("\n")
f.close()
    

