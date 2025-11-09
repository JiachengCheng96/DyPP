# %%
import numpy as np

import torch
from torch import nn
import torchvision
torch.set_printoptions(precision=4,sci_mode=False)

from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

toTensor = torchvision.transforms.ToTensor()
toPIL = T.ToPILImage()

from torch.utils.data import Dataset
from torchvision import datasets

from torch.fft import fft2, ifft2, fftshift, ifftshift, rfft2, irfft2



import platform

# SensorNoise = 1e-3
# SensorNoise = 5e-3
SensorNoise = 3e-3

def normalize_psf(psf, type='sum'):
    assert len(psf.shape) == 3 
    assert psf.shape[1] == psf.shape[2]

    if type == 'max':
        # psf_norm = psf / (psf.view(3,-1).sum(dim=-1)).view(3, 1, 1)
        psf_norm = psf / torch.amax(psf, dim=(-1,-2), keepdim=True)
    elif type == 'sum':
        psf_norm = psf / torch.sum(psf, dim=(-1,-2), keepdim=True)
    else:
        raise ValueError('Normailization type not supported.')

    return psf_norm



def propimg_bchw(psf, img):
    H = img.size(-2)
    W = img.size(-1)
    assert H==W


    if len(img.shape) == 3:
        img = img[None, :, :, ]

    assert len(img.shape) == 4

    psf = normalize_psf(psf, type='sum')

    psf_pad = F.pad(psf, pad=(int(W/2),int(W/2),int(H/2),int(H/2)), mode='constant', value=0)
    # print(psf_pad.size())
    img_pad = F.pad(img, pad=(int(W/2),int(W/2),int(H/2),int(H/2)), mode='constant', value=0)
    # print(img_pad.size())
    
    conv_img = irfft2_bchw(rfft2_bchw(img_pad) * rfft2_bchw(psf_pad), size=img_pad.shape[-2:])

    output_img = conv_img[:,:,int(H/2):int(H/2)+H,int(W/2):int(W/2)+W].clone()

    return output_img


def fft2_bchw(img):
    # img_f = torch.fft.fft2(img)
    img_f = fftshift(fft2(fftshift(img, dim=(-2,-1))), dim=(-2,-1))

    return img_f
    
def ifft2_bchw(img_f):
    # img = torch.fft.ifft2(img_f)
    img = ifftshift(ifft2(ifftshift(img_f, dim=(-2,-1))), dim=(-2,-1))

    return img

# @torch.compile
def rfft2_bchw(img):
    # img_f = fftshift(rfft2(fftshift(img, dim=(-2,-1))), dim=(-2,-1))
    img_f = rfft2(fftshift(img, dim=(-2,-1)))
    # img_f = rfft2(img)

    return img_f

# @torch.compile
def irfft2_bchw(img_f, size=None):
    # img = ifftshift(irfft2(ifftshift(img_f, dim=(-2,-1)), size), dim=(-2,-1))
    img = ifftshift(irfft2(img_f, size), dim=(-2,-1))
    # img = irfft2(img_f, size)

    return img


# Reference 
# https://github.com/polmorenoc/inversegraphics/blob/master/zernike.py

from scipy.special import factorial as fac
def zernike_rad(m, n, rho):
	"""
	Make radial Zernike polynomial on coordinate grid **rho**.

	@param [in] m Radial Zernike index
	@param [in] n Azimuthal Zernike index
	@param [in] rho Radial coordinate grid
	@return Radial polynomial with identical shape as **rho**
	"""
	if (np.mod(n-m, 2) == 1):
		return rho*0.0

	wf = rho*0.0

	for k in range((n-m)//2+1):
		wf += rho**(n-2.0*k) * (-1.0)**k * fac(n-k) / ( fac(k) * fac( (n+m)/2.0 - k ) * fac( (n-m)/2.0 - k ) )

	return wf

def zernike(m, n, rho, phi, norm=True):
	"""
	Calculate Zernike mode (m,n) on grid **rho** and **phi**.

	**rho** and **phi** should be radial and azimuthal coordinate grids of identical shape, respectively.

	@param [in] m Radial Zernike index
	@param [in] n Azimuthal Zernike index
	@param [in] rho Radial coordinate grid
	@param [in] phi Azimuthal coordinate grid
	@param [in] norm Normalize modes to unit variance
	@return Zernike mode (m,n) with identical shape as rho, phi
	@see <http://research.opt.indiana.edu/Library/VSIA/VSIA-2000_taskforce/TOPS4_2.html> and <http://research.opt.indiana.edu/Library/HVO/Handbook.html>.
	"""

	assert (n-m)%2 == 0

	nc = 1.0
	if (norm):
		nc = (2*(n+1)/(1+(m==0)))**0.5
	if (m > 0): return nc*zernike_rad(m, n, rho) * torch.cos(m * phi)
	if (m < 0): return nc*zernike_rad(-m, n, rho) * torch.sin(-m * phi)
	return nc*zernike_rad(0, n, rho)

def noll_to_zern(j):
	"""
	Convert linear Noll index to tuple of Zernike indices.

	j is the linear Noll coordinate, n is the radial Zernike index and m is the azimuthal Zernike index.

	@param [in] j Zernike mode Noll index
	@return (n, m) tuple of Zernike indices
	@see <https://oeis.org/A176988>.
	"""
	if (j == 0):
		raise ValueError("Noll indices start at 1, 0 is invalid.")

	n = 0
	j1 = j-1
	while (j1 > n):
		n += 1
		j1 -= n

	m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))
	return (n, m)

def zernikel(j, rho, phi, norm=True):
	n, m = noll_to_zern(j)
	return zernike(m, n, rho, phi, norm)



class ZernikeSystem2(nn.Module):
    def __init__(self, img_sz=960, PSF_HorizontalFlip=None):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()


        self.img_sz = img_sz

        # Wvl = torch.tensor([460, 550, 640]) * 1e-9 # wavelength
        Wvl = torch.tensor([640, 550, 460]) * 1e-9 # wavelength

        self.register_buffer('Wvl', Wvl)



        self.Det_PS = 1e-6 # Detector pixel size
        self.NA_Act = 0.12 # Numerical aperture (F# can transfer to NA)

        self.psf_quantize = False


        apert = self.genapert_rgb(self.img_sz, self.img_sz)
        self.register_buffer('apert', apert)

        self.num_zern_coeff = 350
        self.PSF_HorizontalFlip = PSF_HorizontalFlip 

        zern_poly_RGB = torch.zeros([3, self.num_zern_coeff, self.img_sz, self.img_sz])

        with torch.no_grad():
            for i in range(3):
                # pupil size (diameter)
                apert_diameter = torch.sum(self.apert[i, int(self.img_sz/2), :])
                apert_diameter = int(apert_diameter + torch.remainder(apert_diameter, 2))

                xx = torch.linspace(-1, 1, apert_diameter)
                yy = torch.linspace(-1, 1, apert_diameter)

                [X, Y] = torch.meshgrid(xx, yy,indexing='xy')

                rho = torch.sqrt(X**2+Y**2)
                phi = torch.atan2(Y,X)


                zern_poly = torch.zeros([self.num_zern_coeff, apert_diameter, apert_diameter])

                for j in range(self.num_zern_coeff):
                    zern_poly[j,:,:] = zernikel(j+2, rho, phi, norm=True)



                pad_width = int((self.img_sz - apert_diameter) / 2)
                zern_poly_RGB[i, :, :, :] = F.pad(input=zern_poly, pad=(pad_width, pad_width, pad_width, pad_width), mode='constant', value=0)

            self.register_buffer('zern_poly_RGB', zern_poly_RGB)

    def get_device(self):
        return self.Wvl.device


    def genapert_rgb(self, Dim1, Dim2):
        Apert = torch.zeros([3, Dim1, Dim2], device=self.get_device())
        Img_PS = self.Det_PS
        Dim1_w2 = 2*np.pi/Img_PS
        
        for ii, wvl_ii in enumerate(self.Wvl):

            Rad2 = 2 * np.pi * self.NA_Act/ wvl_ii
            Fxx = torch.linspace(-Dim1_w2/2, Dim1_w2/2, Dim1, device=self.get_device())
            Fyy = torch.linspace(-Dim1_w2/2, Dim1_w2/2, Dim2, device=self.get_device())

            [Fr_X, Fr_Y] = torch.meshgrid(Fxx, Fyy, indexing='ij')

            Apert[ii,:,:] = torch.sqrt(Fr_X**2+Fr_Y**2) <= Rad2
        return Apert

    
    def otf2psf(self, otf):
        psf = torch.abs(ifft2_bchw(otf))**2

        return psf
        



    def get_ZernikeP_RGB(self, z):


        Zernike_sum = z[None,:,None,None] * self.zern_poly_RGB[:,:,:,:]
        Zernike_sum = Zernike_sum.sum(dim=1)

        ZernikeP = torch.rot90(torch.exp(1j*2*np.pi*Zernike_sum), 2, [1,2])

        # print(f'ZernikeP: {ZernikeP.dtype}')

        assert ZernikeP.shape[-1] == ZernikeP.shape[-2]
        return ZernikeP
    
    def get_otf(self, z):


        ZernikeP = self.get_ZernikeP_RGB(z) * self.apert[:,:,:] + 1e-19

        # print(ZernikeP.shape)
        return ZernikeP
    
    def get_psf(self, z, normalize='sum'):

        z = z.squeeze()

        otf = self.get_otf(z)
        psf = self.otf2psf(otf)

        psf_flip = torch.flip(psf, dims=[-1])
        if self.PSF_HorizontalFlip is True:
            psf = psf_flip
        elif self.PSF_HorizontalFlip is None:
            if np.random.randn()>0:
                psf = psf_flip
        
        if self.psf_quantize:
             psf = psf

        psf = normalize_psf(psf, type=normalize)

        return psf

    def forward(self, x, z):
        if len(x.shape) == 3:
            # CHW -> BCHW
            x = torch.unsqueeze(x, 0)
        
        assert len(x.shape) == 4

        H, W = x.shape[-2:]

        assert (H, W) == (self.img_sz, self.img_sz), "input size not supported!"

        psf = self.get_psf(z)

        output = propimg_bchw(psf, x)


        return output
    



class CameraModel(nn.Module):
    def __init__(self, img_sz=960, dh=0, dw=0):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()

        zernike_sys = ZernikeSystem2(img_sz=img_sz)

        self.img_sz = img_sz
        self.zernike_sys = zernike_sys

        self.dh = dh
        self.dw = dw

        self.debug=False

        
    def forward(self, x, z):
        if len(x.shape) == 3:
            # CHW -> BCHW
            x = torch.unsqueeze(x, 0)
        
        assert len(x.shape) == 4

        H, W = x.shape[-2:]
        assert (H, W) == (self.img_sz, self.img_sz)

        output = self.zernike_sys(x, z)

        # sensor noise
        output += SensorNoise * torch.randn_like(output)
        output = torch.clamp(output, 0.0, 1.0)

        if self.debug and x.shape[0] == 1:
            print('debug mode on')

            torchvision.utils.save_image(x, '/mnt/data/temp/input_ours.png')
            torchvision.utils.save_image(output, '/mnt/data/temp/output_ours.png')

            print(output.shape, x.shape)

        output_crop = output[:, :, self.dh : H-self.dh, self.dw : W-self.dw].clone()
        input_crop = x[:, :, self.dh : H-self.dh, self.dw : W-self.dw].clone()

        del output, x
        return output_crop, input_crop

