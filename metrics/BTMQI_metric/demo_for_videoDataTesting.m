
%==========================================================================
% 1) Please cite the paper (K. Gu, S. Wang, G. Zhai, S. Ma, X. Yang, W. Lin, 
% W. Zhang, and W. Gao, "Blind quality assessment of tone-mapped images via 
% analysis of information, naturalness and structure," IEEE Trans. Multimedia,
% vol. 18, no. 3, pp. 432-443, Mar. 2016.)
% 2) If any question, please contact me through guke.doctor@gmail.com; 
% gukesjtuee@gmail.com. 
% 3) Welcome to cooperation, and I am very willing to share my experience.
%==========================================================================

clear;
clc;

score_avg = 0
num = 0
main_dir = '..\\output_ours'
scenes = dir(main_dir)
for k=3:22
    paths = dir([main_dir,'\\',scenes(k).name])
    for i=3:8
        path = [main_dir,'\\',scenes(k).name,'\\',paths(i).name]
        im = imread(path);
        im = (im-37.5)*0.9+45;
        imshow(im)
        [score,feature] = BTMQI(im);
        score
        score_avg = score_avg + score;
        num = num + 1;
    end
end
score_avg = score_avg/num