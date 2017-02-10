-- Bookstein, F. L. 
-- "Principal Warps: Thin Plate Splines and the Decomposition of Deformations."
-- IEEE Trans. Pattern Anal. Mach. Intell. 11, 567-585, 1989. 

require "image"



utils = {}

function utils.meshgrid(x, y)
    local xx = torch.repeatTensor(x, y:size(1),1)
    local yy = torch.repeatTensor(y:view(-1,1), 1, x:size(1))
    return xx, yy
end

warp2d = {}


function warp2d.warp(im_in, pts_in, pts_def, method)
    

    local im_size = im_in:size()

    local imc = im_size[1]
    local imh = im_size[2]
    local imw = im_size[3]
    
    local r = 0.1*imw
    
    local im_out  = torch.Tensor(imc, imh, imw):fill(0)
    local im_mask = torch.Tensor(imc, imh, imw):fill(0)
    
    local num_pts = pts_in:size(1)
    
    
    local K = torch.Tensor(num_pts, num_pts):fill(0)
    
    for i = 1,num_pts do
        local pt = pts_in[{i,{}}]
        pt:resize(pt:size(1),1):size()
        
        local pts = pt:t():repeatTensor(num_pts,1)
        local dx = pts - pts_def
        K[{{},{i}}] = torch.sum(torch.pow(dx, 2),2)
    end    
    local K = warp2d.rbf(K,r)
    
    local P = torch.cat(torch.Tensor(num_pts,1):fill(1), pts_def, 2)
    
    local L = torch.cat(torch.cat(K,P,2),  torch.cat(P:t(), torch.Tensor(3,3):fill(0),2), 1)

    local Y = torch.cat(pts_in, torch.Tensor(3,2):fill(0), 1)
    
    -- L_inv = torch.potri(torch.potrf(L))
    local L_inv = torch.inverse(L)
    
    local w = torch.Tensor(Y:size()):fill(0):addmm(L_inv, Y)
    
    local x, y = utils.meshgrid(torch.linspace(1,imw, imw), torch.linspace(1,imh, imh))
    local pt_xy = torch.cat(torch.reshape(x, imw * imh,1 ),torch.reshape(y, imw*imh, 1),2)
    
    local nump = pt_xy:size()[1]
    local Kp = torch.Tensor(nump, num_pts):fill(0)
    
    
    for i=1,num_pts do
        local pt = pts_in[{i,{}}]
        pt:resize(1,pt:size(1)):size()
        
        local dx = torch.Tensor(nump, 2):fill(0):addmm(torch.Tensor(nump,1):fill(1), pt)
        local dx = torch.add(dx, -pt_xy)
        
        Kp[{{},{i}}] = torch.sum(torch.pow(dx,2),2)
    end
    local Kp = warp2d.rbf(Kp, r)
    
    local L = torch.cat(torch.cat(Kp, torch.Tensor(nump,1):fill(1)), pt_xy)
    
    local pt_all = torch.Tensor(nump,2):fill(0):addmm(L, w)
    
    -- print(pt_all[{{1},{}}])
    -- print(pt_all)
    
    local xd = torch.reshape(pt_all[{{},{1}}], 1, imh, imw) 
    local yd = torch.reshape(pt_all[{{},{2}}], 1, imh, imw) 
    
    local coords = torch.cat(xd,yd, 1)

    local x = torch.reshape(x, 1, imh, imw)
    local y = torch.reshape(y, 1, imh, imw)
    
    local coords_orig = torch.cat(x,y,1)
    
    local warpfield = torch.add(coords,-coords_orig)
    
    local img_out = image.warp(im_in, warpfield)
    return img_out, warpfield    
end

function warp2d.gen_warp_pts(imsize, ndefpts, stdev_factor)
    -- figure how the grid spacing for the deformation points
    local x_pt_spacing = imsize[2]/ndefpts
    local y_pt_spacing = imsize[3]/ndefpts

    local stdev = x_pt_spacing/stdev_factor

    local xpts = torch.cat(torch.range(1,imsize[2], x_pt_spacing), torch.Tensor(1):fill(imsize[2]))
    local ypts = torch.cat(torch.range(1,imsize[3], y_pt_spacing), torch.Tensor(1):fill(imsize[3]))
    -- ypts = torch.range(1,imsize[3], imsize[3]/4)

    local xpts_grd, ypts_grd = utils.meshgrid(xpts, ypts)

    local sel_mask = torch.ByteTensor(xpts_grd:size()):fill(0)
    sel_mask[{{2,-2}, {2,-2}}] = 1

    local x_defpts_fixed = xpts_grd:maskedSelect(sel_mask)
    local y_defpts_fixed = ypts_grd:maskedSelect(sel_mask)

    local defpts_fixed = torch.cat(x_defpts_fixed, y_defpts_fixed, 2)

    local x_anchor_pts = xpts_grd:maskedSelect(torch.eq(sel_mask, 0))
    local y_anchor_pts = ypts_grd:maskedSelect(torch.eq(sel_mask, 0))

    local anchor_pts = torch.cat(x_anchor_pts, y_anchor_pts,2)

    local defpts = torch.Tensor(defpts_fixed:size()):copy(defpts_fixed)

    for i= 1,defpts:size()[1] do
        defpts[{{i},{}}] = defpts[{{i},{}}] + torch.randn(2)*stdev
    end

    local pts_anchor = torch.cat(anchor_pts, defpts_fixed, 1)
    local pts_def = torch.cat(anchor_pts, defpts, 1)

    return pts_anchor, pts_def
end

function warp2d.meshgrid(x, y)
    local xx = torch.repeatTensor(x, y:size(1),1)
    local yy = torch.repeatTensor(y:view(-1,1), 1, x:size(1))
    return xx, yy
end

function warp2d.rbf(d, r)
    local ko = -d/torch.pow(r,2)
    local ko = ko:exp()

    return ko
end

