% run reg_nifti

li = '----------';
li = strcat(li,li,li,li,li,li,li, '\n');
FIXED = 'adc';
MOVING = {'bval','ktrans'};
%%
trainIDS;
pids = unique(ProxID);

fprintf(li);
fprintf('Registering Training Data\n');
fprintf('FIXED: %s\n', fixed)
fprintf('MOVING: %s %s\n', moving{1}, moving{2})
fprintf(li);

header = 'pid \t\t\t\t time\n';
fprintf(header);
fprintf(li);
start = tic;
for j = 1:size(pids) - 3;
    tic;
    pid = pids(j);
    pid = pid{1};
    fprintf(pid);
    reg_nifti(FIXED, MOVING, 'traindata', pid);
    
    elapsed = toc;
    fprintf('\t\t\t %f\r', elapsed);
    
end
stop = toc(start);
fprintf(li);
fprintf('Total Time Elapsed: %f \n', stop);
fprintf(li);

fprintf('\n');
fprintf('\n');

%%
% Testing Images
testIDS;
pids = unique(ProxID);

fprintf(li);
fprintf('Registering Test Data\n');
fprintf(li);

header = 'pid \t\t\t\t time\n';
fprintf(header);
fprintf(li);
start = tic;
for j = 12:size(pids);
    tic;
    pid = pids(j);
    pid = pid{1};
    fprintf(pid);
    reg_nifti(FIXED, MOVING, 'testdata', pid);
    
    elapsed = toc;
    fprintf('\t\t\t %f\r', elapsed);
   
end
stop = toc(start);
fprintf(li);
fprintf('Total Time Elapsed: %f \n', stop);
fprintf(li);