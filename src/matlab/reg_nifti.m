% NIFTI image registration
% Author: Morten Vester Pedersen
function [fixed, moving] = reg_nifti(fixed_t, moving_t, train, pid)

outdir = strcat('/Users/vester/git/tcia-challenge/',train,'/nifti');

% Loading fixed image
f_path = strjoin({outdir, fixed_t, pid, strcat(pid, '.nii')},'/');
fixed = load_untouch_nii(f_path);
fixed = double(fixed.img);
fixed = fixed / sum(fixed(:));

% Multimodal since we can have differnt image types
[optimizer, metric] = imregconfig('multimodal');

% Options
optimizer.InitialRadius = 0.0001;
optimizer.Epsilon = 1.5e-4;
optimizer.GrowthFactor = 1.01;
optimizer.MaximumIterations = 5000;

for mt = moving_t
    % load moving img
    m_path = strjoin({outdir, mt{1}, pid, strcat(pid, '.nii')},'/');
    moving = load_untouch_nii(m_path);
    moving = double(moving.img);
    moving = moving / sum(moving(:));
    sz = size(moving);
    
    % Register Image
    if sz(3) < 16;
        % Setup intial transform
        tform = imregtform(moving, fixed, 'affine', ...
                           optimizer, metric, ...
                           'PyramidLevels',2);
        
        %movingRegistered = imwarp(moving,tform,'OutputView',imref2d(size(fixed)));
        registered = imregister(moving, fixed, 'affine',...
                                optimizer, metric, ...
                                'PyramidLevels',2,...
                                'InitialTransformation', tform);
    else
        tform = imregtform(moving, fixed, 'affine', optimizer, metric);
        %movingRegistered = imwarp(moving,tform,'OutputView',imref2d(size(fixed)));
        registered = imregister(moving, fixed, 'affine',...
                                optimizer, metric, ...
                                'InitialTransformation', tform,...
                                'PyramidLevels',3);
    end
    
    o_path = strjoin({outdir, 'registered', strcat(pid, mt{1}, '.nii')},'/');
    save_nii(make_nii(registered), o_path);
end

o_path = strjoin({outdir, 'registered', strcat(pid, fixed_t, '.nii')},'/');
save_nii(make_nii(fixed), o_path);

end