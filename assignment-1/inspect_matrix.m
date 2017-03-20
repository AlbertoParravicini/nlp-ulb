%% 
occ_matrix_gb = load('-mat', 'language_model_occurrencies_GB');
vars = fieldnames(occ_matrix_gb);
for i = 1:length(vars)
    assignin('base', vars{i}, occ_matrix_gb.(vars{i}));
end

vocabulary = 'abcdefghijklmnopkrstuvwxyz_'

for i=1:27
    subplot(4, 7, i)
    spy(occ_matrix_gb(:, :, i))
    title(vocabulary(i))
end



%%
clc 
clear all
close all

importfile('language_model_occ_GB_small');
importfile('language_model_occ_US_small');
importfile('language_model_occ_AU_small');

%%
diff_gb_us = abs(occ_matrix_gb_small - occ_matrix_us_small);
diff_gb_us = diff_gb_us - 5 * mean(nonzeros(diff_gb_us));
diff_gb_us(diff_gb_us<0)=0;
vocabulary = 'abcdefghijklmnopkrstuvwxyz_';

for i=1:27
    subplot(4, 7, i)
    spy(diff_gb_us(:, :, i))
    title(vocabulary(i))
end

%%
spy(diff_gb_us(:, :, 14))
grid on

% DIFF GB US: -ion
