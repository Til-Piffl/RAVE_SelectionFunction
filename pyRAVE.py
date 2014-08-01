import numpy as np
import healpy
import cPickle

STR_TAGS = ['rave_obs_id','raveid','id_2mass','id_denis','id_ucac4',\
            'id_ppmxl','c1','c2','c3','c4','c5','c6','zeropointflag']

# Transformation matrix between equatorial and galactic coordinates:
# [x_G  y_G  z_G] = [x  y  z] . A_G
A_G = np.array([[-0.0548755604, +0.4941094279, -0.8676661490],
                [-0.8734370902, -0.4448296300, -0.1980763734],
                [-0.4838350155, +0.7469822445, +0.4559837762]])

# inverse matrix
iA_G = np.linalg.inv(A_G)
d2r = np.pi/180
r2d = 180./np.pi
def gal2eqCoords(L,B):
    '''
    Computes Equatorial angular coordinates via the method descibed
    in the Introduction to the Hipparcos catalog.

    Input:

    L    - Galactic longitude in degree (1d-array or single)
    B    - Galactic latitude in degree (1d-array or single)

    Output:

    RA   - right ascension in degree
    DE   - declination in degree
    
    History:
    Written by Til Piffl                          May 2013
    '''

    l = np.array(L).reshape(-1)*d2r
    b = np.array(B).reshape(-1)*d2r
    assert len(l) == len(b)
    sb,cb = np.sin(b),np.cos(b)
    sl,cl = np.sin(l),np.cos(l)
    
    aux0 = iA_G[0,0]*cb*cl + iA_G[1,0]*cb*sl + iA_G[2,0]*sb
    aux1 = iA_G[0,1]*cb*cl + iA_G[1,1]*cb*sl + iA_G[2,1]*sb
    aux2 = iA_G[0,2]*cb*cl + iA_G[1,2]*cb*sl + iA_G[2,2]*sb

    de = np.arcsin(aux2)*r2d
    ra = np.arctan2(aux1,aux0)*r2d
    ra[ra<0] += 360.

    if len(ra) == 1:
        return ra[0],de[0]
    else:
        return ra,de

def eq2galCoords(RA,DE):
    '''
    Computes Galactic angular coordinates via the method descibed
    in the Introduction to the Hipparcos catalog.

    Input:

    RA   - right ascension in degree (1d-array or single)
    DE   - declination in degree (1d-array or single)

    Output:

    l    - Galactic longitude in degree
    b    - Galactic latitude in degree
    
    History:
    Written by Til Piffl                          May 2013
    '''

    ra = np.array(RA).reshape(-1)*d2r
    de = np.array(DE).reshape(-1)*d2r
    assert len(ra) == len(de)

    sde,cde = np.sin(de),np.cos(de)
    sra,cra = np.sin(ra),np.cos(ra)
    
    aux0 = A_G[0,0]*cde*cra + A_G[1,0]*cde*sra + A_G[2,0]*sde
    aux1 = A_G[0,1]*cde*cra + A_G[1,1]*cde*sra + A_G[2,1]*sde
    aux2 = A_G[0,2]*cde*cra + A_G[1,2]*cde*sra + A_G[2,2]*sde

    b = np.arcsin(aux2)*r2d
    l = np.arctan2(aux1,aux0)*r2d
    l[l<0] += 360.

    if len(l) == 1:
        return l[0],b[0]
    else:
        return l,b



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
    l0 = [tag.lower()[1:-1] for tag in f.readline()[:-1].split(',')]
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

def apply_footprint(l,b,ra,dec):
    '''
    -----------------
    in_footprint = pyRAVE.apply_footprint(l,b,ra,dec)
    -----------------
    Applies RAVE footprint and removes regions on the sky that are
    not suited for statistical analysis, mainly calibration fields
    and from the intermediate input catalogue.

    -----------------
    Input

    l :      Galactic longitude; 1d array
    b :      Galactic latitude; 1d array
    ra :     Right ascension; 1d array
    dec :    Declination; 1d array

    -----------------
    Output

    in_footprint :  1d bool array of same length as input
                            arrays. 'True' for stars inside the
                            footprint, 'False' otherwise.
    -----------------
    '''

    L = np.copy(l)
    if (L<0).any():
        L[L<0] += 360

    visible_sky = (dec < 5)
    infootprint = (abs(b) >= 25) | \
                  ( (abs(b)>=5)&(L>=225)&(L<=315) ) | \
                  ( (b>10)&(b<25)&(L<=330)&(L>=30) ) | \
                  ( (b<-10)&(b>-25) )
    twomass_ext = (dec <= 2) | \
                  ( (dec<5)&(ra <= 90) ) |\
                  ( (dec<5)&(ra >= 112.5)&(ra <= 255) ) |\
                  ( (dec<5)&(ra >= 292.5) )
    return infootprint & visible_sky & twomass_ext


def computeHEALPIX_ids(l,b):
    '''
    ----------------
    ids = pyRAVE.computeHEALPIX_ids(l,b)
    ----------------

    Input
    
    l :      Galactic longitude in degree; 1d array
    b :      Galactic latitude in degree; 1d array

    Output

    ids :     HEALPIX ids (1d array) for NSIDE = 32
    '''
    NSIDE = 32 # Fixed in precomputed 2MASS table!
    return healpy.ang2pix(NSIDE,(90.-b)*np.pi/180,l*np.pi/180.)


def computeCompleteness(HEALPIX_ids,I2MASS):
    '''
    -----------------------
    comp,Ibins = pyRAVE.computeCompleteness(l,b,I2MASS)
    -----------------------
    Returns completeness w.r.t. 2MASS in (l,b,I2MASS)-bins. The (l,b) grid
    is done using HEALPIX pixelisation of the sky (NSIDE = 32). This results
    in 12288 pixels. The bin width in the I magnitudes is 0.2 mag.

    -----------------------
    Input

    HEALPIX_ids : HEALPIX ids for NSIDE = 32 (12288 pixels); 1d array
    I2MASS :      approximated I magnitude; 1d array

    -----------------------
    Output

    comp  :      2d array with completeness values (Number of
                 pixels x Number of I magnitude bins)
    Ibins :      I magnitude bin borders

    -----------------------
    '''

    fname = '2MASS_number_counts.txt'
    f = open(fname,'r')
    l = f.readlines()
    NSIDE = int(l[0].split()[0])
    Npix = healpy.nside2npix(NSIDE)
    NIbins = int(l[0].split()[1])
    Ibins = np.array([float(i) for i in l[1].split()])
    f.close()
    dist2MASS = np.loadtxt(fname,skiprows=2)

    hp_ids = np.copy(HEALPIX_ids)
    RAVEorder = np.argsort(hp_ids)
    i2mass = np.copy(I2MASS)

    NR = 1.*np.histogram(hp_ids,np.arange(Npix+1))[0]

    comp = np.zeros((Npix,len(Ibins)-1))

    count = 0
    countS = 0
    for i in range(Npix):
        if NR[i] == 0:
            continue
        subRAVE  = RAVEorder[np.sum(NR[:i]):np.sum(NR[:i+1])]

        distRAVE  = 1.*np.histogram(i2mass[subRAVE],Ibins)[0]
        if not (dist2MASS[i] >= distRAVE).all():
            count += 1
            take = dist2MASS[i] < distRAVE
            print 'In pixel %i: more rave than 2MASS stars'%(i), \
                np.sum(distRAVE[take]),np.sum(dist2MASS[i][take]), \
                90.-healpy.pix2ang(NSIDE,i)[0]*180./np.pi
            countS += np.sum(distRAVE[take])-np.sum(dist2MASS[i][take])
            distRAVE[take] = dist2MASS[i][take]
        tmp = dist2MASS[i] != 0
        comp[i][tmp] = distRAVE[tmp]/dist2MASS[i][tmp]

    print count, countS
    return comp,Ibins
