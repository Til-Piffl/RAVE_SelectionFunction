import numpy as np
import healpy
import cPickle

STR_TAGS = ['rave_obs_id','raveid','id_2mass','id_denis','id_ucac4',\
            'id_ppmxl','c1','c2','c3','c4','c5','c6','zeropointflag']

def readCSV(fname,str_tags=STR_TAGS):
    '''
    ----------------
    RAVE = pyRAVE.readCSV(fname[,str_tags])
    ----------------
    Reads in a csv-file as provided by the RAVE data base on
    www.rave-survey.de
    
    ----------------
    Input

    fname:       file name of the csv file
    str_tags:    optional; list of column names that contain string
                 entries that should or can not be converted into floats;
                 defaults lists a number of common column names.

    ----------------
    Output

    RAVE:        dictionnary with entries for each column in the csv-file;
                 Entries keys are determined by the first line in the csv-file

    ----------------
    '''
    f = open(fname,'r')
    l0 = [tag.lower() for tag in f.readline()[:-1].split(',')]
    l = f.readlines()
    Nentries =len(l)
    f.close()

    RAVE = {}
    for tag in l0:
        RAVE[tag] = []

    for i in range(Nentries):
        L = l[i].split(',')
        for j,tag in enumerate(l0):
            if tag.lower() not in str_tags:
                if ((L[j] == '') or (L[j] == '\n')):
                    RAVE[tag].append(np.NaN)
                else:
                    RAVE[tag].append(float(L[j]))
            else:
                RAVE[tag].append(L[j])
    for tag in l0:
        RAVE[tag] = np.array(RAVE[tag])

    return RAVE

def computeI2MASS(Jmag,Kmag):
    '''
    --------------
    I2MASS = pyRAVE.computeI2MASS(Jmag,Kmag)
    --------------
    Computes an approximated I magnitude from 2MASS J and Ks.

    --------------
    Input

    Jmag:      2MASS J magnitudes; 1d array
    Kmag:      2MASS Ks magnitudes; 1d array

    --------------
    Output

    I2MASS:    approximated I magnitude; 1d array
  
    '''
    I2MASS = lambda J, K: J + 0.12 + (J-K) + 0.2*np.exp(((J-K)-1.2)/0.2)
    return I2MASS(Jmag,Kmag)

def findUnique(raveid,rave_obs_id,snr):
    '''
    --------------
    unique = pyRAVE.findUnique(raveid,rave_obs_id,snr)
    --------------
    Finds unique RAVE targets in the provided catalogue. For multible
    observations the one with the highest signal-to-noise is selected.

    --------------
    Input:
    
    raveid:       1d string array.
    rave_obs_id:  1d string array of the same length as 'raveid' and
                  with the same ordering.
    snr:          S/N of the observations; 1d float array of the same
                  length as 'raveid' and with the same ordering.
    
    --------------
    Output:

    unique:       1d bool array of the same length and ordering as 'raveid';
                  'True' for single observations or observations with the
                  highest S/N value if a target was multibly observerd. 'False'
                  otherwise.

    --------------
    
    '''
    assert len(rave_obs_id) == len(raveid)
    assert len(raveid) == len(snr)

    Nentries = len(rave_obs_id)
    IDorder = np.argsort(raveid)
    unique = np.zeros(Nentries,dtype=bool)
    SNR = snr[IDorder]
    RAVEID = raveid[IDorder]
    unique[IDorder[0]] = True
    i0 = 0
    for i in xrange(1,Nentries):
        if RAVEID[i] == RAVEID[i0]:
            if SNR[i] > SNR[i0]:
                unique[IDorder[i]] = True
                unique[IDorder[i0]] = False
                i0 = i
        else:
            unique[IDorder[i]] = True
            i0 = i
    #print "Found %i unique entries."%(np.sum(unique))
    return unique



def JmK_color_cut(b,Jmag,Kmag):
    '''
    -----------------
    keep = pyRAVE.JmK_color_cut(b,Jmag,Kmag)
    -----------------
    Enforces the J-Ks > 0.5 mag colour cut close to the Galactic plane
    and the Bulge region.

    -----------------
    Input

    b :         Galactic latitude in degree; 1d array
    Jmag :      2MASS J magnitude; 1d array
    Kmag :      2MASS Ks magnitude; 1d array

    -----------------
    Output

    keep :      1d bool array of same length as input arrays; 'True' for
                entries that are consistent with the colour criterion, 'False'
                otherwise.

    -----------------
    '''
    assert len(b) == len(Jmag)
    assert len(b) == len(Kmag)
    JmK = Jmag-Kmag
    keep = (abs(b) >= 25) | (JmK >= 0.5)
    return keep


def remove_problematic_fields(l,b):
    '''
    -----------------
    unproblematic_fields = pyRAVE.remove_problematic_fields(l,b)
    -----------------
    Removes regions on the sky that are not suited for statistical
    analysis, mainly calibration fields and from the intermediate
    input catalogue.

    -----------------
    Input

    l :      Galactic longitude; 1d array
    b :      Galactic latitude; 1d array

    -----------------
    Output

    unproblematic_fields :  1d bool array of same length as input
                            arrays. 'True' for stars in unproblematic
                            regions, 'False' otherwise.
    -----------------
    '''
    problematic_fields =  (abs(b) < 5) |\
                          ((abs(b) < 10.)&((l<45.)|(l>315.))) |\
                          (((l>330)|(l<30))&(b<25)&(b>0))
    return problematic_fields == False


def computeHEALPIX_ids(ra,de):
    '''
    ----------------
    ids = pyRAVE.computeHEALPIX_ids(ra,de)
    ----------------

    Input
    
    ra :      Right ascension in degree; 1d array
    de :      Declination in degree; 1d array

    Output

    ids :     HEALPIX ids (1d array) for NSIDE = 32
    '''
    NSIDE = 32 # Fixed in precomputed 2MASS table!
    return healpy.ang2pix(NSIDE,(90.-de)*np.pi/180,ra*np.pi/180.)


def computeCompleteness(HEALPIX_ids,I2MASS,Jmag,Kmag,dI):
    '''
    -----------------------
    comp = pyRAVE.computeCompleteness(l,b,I2MASS)
    -----------------------
    Returns completeness w.r.t. 2MASS in (ra,dec,I2MASS)-bins. The (ra,dec) grid
    is done using HEALPIX pixelisation of the sky (NSIDE = 32).

    -----------------------
    Input

    HEALPIX_ids : HEALPIX ids for NSIDE = 32; 1d array
    I2MASS :      approximated I magnitude; 1d array
    Jmag:         2MASS J magnitude; 1d array
    Kmag:         2MASS Ks magnitude; 1d array

    dI :          Bin width for the I2mass grid; float

    -----------------------
    Output

    comp ....

    -----------------------
    '''

    fname = 'tmp2MASS.dat'
    f = open(fname,'rb')
    NSIDE = cPickle.load(f)
    I_limits = cPickle.load(f) # [8,13]
    tmp_Jrange = cPickle.load(f) # [6,13.5]
    tmp_Krange = cPickle.load(f) # [5,13]
    tmp_JmK_min = cPickle.load(f)
    TwoMASS = {}
    TwoMASS['Imag'] = cPickle.load(f)
    TwoMASS['healpixNum'] = cPickle.load(f)
    N2M = cPickle.load(f)
    TwoMASSorder = cPickle.load(f)
    f.close()

    Limits_2MASS = (Jmag >= tmp_Jrange[0]) & (Jmag <= tmp_Jrange[1]) & \
                   (Kmag >= tmp_Krange[0]) & (Kmag <= tmp_Krange[1]) & \
                   (I2MASS >= I_limits[0]) & (I2MASS <= I_limits[-1])


    hp_ids = HEALPIX_ids[Limits_2MASS]
    RAVEorder = np.argsort(hp_ids)

    i2mass = I2MASS[Limits_2MASS]

    Ibins = np.arange(I_limits[0],I_limits[1]+dI,dI)
    Npix = healpy.nside2npix(NSIDE)
    NR = 1.*np.histogram(hp_ids,np.arange(Npix+1))[0]

    comp = np.zeros((Npix,len(Ibins)-1))

    for i in range(Npix):
        if NR[i] == 0:
            continue
        if i == 0:
            subRAVE  = RAVEorder[:NR[i]]
            sub2MASS  = TwoMASSorder[:N2M[i]]
        else:
            subRAVE  = RAVEorder[np.sum(NR[:i-1]):np.sum(NR[:i])]
            sub2MASS  = TwoMASSorder[np.sum(N2M[:i-1]):np.sum(N2M[:i])]

        distRAVE  = 1.*np.histogram(i2mass[subRAVE],Ibins)[0]
        dist2MASS = 1.*np.histogram(TwoMASS['Imag'][sub2MASS],Ibins)[0]
        dist2MASS[distRAVE>dist2MASS] = distRAVE[distRAVE>dist2MASS]
        if not (dist2MASS >= distRAVE).all():
            print i, 'more rave than 2mass stars', \
                np.sum(distRAVE),np.sum(dist2MASS), \
                90.-healpy.pix2ang(NSIDE,i)[0]*180./pi
            print ['%.1f'%i for i in distRAVE]
            print ['%.1f'%i for i in (distRAVE/dist2MASS)]
            break
        tmp = dist2MASS != 0
        comp[i][tmp] = distRAVE[tmp]/dist2MASS[tmp]

    return comp,Ibins
