function main
  global T M Q O
  T = 89;         %Number of vectors in a sequence 
  M = 3;          %Number of mixtures 
  Q = 3;          %Number of states 
%%%%%%%%%%%%%% train %%%%%%%%%%%%%%
   % train 14 dims together
%    O = 14;          %Number of coefficients in a vector 
%    for i = 0:2
%        models(i+1,:).m = trainAllDim(i); 
%    end
    
%    train each dim
%    O = 1;
%    for i = 0:2
%        fprintf('class: %d\n', i+1);
%        models(i+1,:) = trainEachDim(i); 
%    end
%    save('models.mat', 'models');
%%%%%%%%%%%%%% test %%%%%%%%%%%%%%
[p,confMatrix] = test();
p
confMatrix
% %% AdaBoost
% weakTrainData();
end
function [per,confMatrix] = test()
   load models;
   count = 0;
   total_count = 0;
   confMatrix = zeros(3,3);
   for i = 1 : 3
       testData = getData('test', i-1);
       for n = 1 : size(testData, 3)
           for dim = 1 : 14
               for j = 1 : 3
                gmm = models(j, dim);
                data = testData(dim,:,n);
                loglik(j, dim) = mhmm_logprob(data ,gmm.prior, gmm.transmat,gmm.mu, gmm.sigma, gmm.mixmat);
               end
                x = loglik(:,dim);
                c = find(x==max(x));               
                if size(c,1) > 1
                    result(dim,1) = randi([1 size(c,1)],1,1);
                else
                    result(dim,1) = c;
                end
                result(dim,2) = max(x);                
           end
           if (i == mode(result(:,1)))
               count  = count + 1;
           end
           confMatrix(i,mode(result(:,1))) = confMatrix(i,mode(result(:,1))) + 1; 
           total_count = total_count + 1;
           llm(i,n).m = loglik;
       end  
   end
    per =  count / total_count;
    save('llm.mat', 'llm');
end

function data = getDataShort(mode, class)
    if strcmp(mode, 'test')
        s = what(['/Users/zhangchi8620/Codes/HMMall/data/class',int2str(class),'_test/']); %look in current directory
    elseif strcmp(mode, 'train')
        s = what(['/Users/zhangchi8620/Codes/HMMall/data/class',int2str(class), '/']); %look in current directory
    else
        return;    
    end
    matfiles=s.mat;
    maxLen = 0;
    data = [];
    % count = 0;
    for a=1:numel(matfiles)
        tmp=load([char(matfiles(a))]);
        tmp = tmp.filedata;
        if size(tmp,1) > maxLen
            maxLen = size(tmp,1);
        end
        data(:,:,a) = [tmp(1:89,:)]';
    end
end

function data = getData(mode, class)
    if strcmp(mode, 'test')
        s = what(['/Users/zhangchi8620/Codes/HMMall/data/class',int2str(class),'_test']); %look in current directory
    elseif strcmp(mode, 'train')
        s = what(['/Users/zhangchi8620/Codes/HMMall/data/class',int2str(class)]); %look in current directory
    else
        return;    
    end
    matfiles=s.mat;
    maxLen = 0;
    data = [];
    % count = 0;
    for a=1:numel(matfiles)
        tmp=load(char(matfiles(a)));
        tmp = tmp.filedata;
        if size(tmp,1) > maxLen
            maxLen = size(tmp,1);
        end
    end
    
    data = zeros(14, maxLen, numel(matfiles));
    for a=1:numel(matfiles)
        tmp=load(char(matfiles(a)));
        tmp = tmp.filedata;
        data(:,1:size(tmp,1),a) = tmp';
    end
    
end


function gmm = trainAllDim(class)
    global T M Q O
    tmpdata = getData('train', class);
    
    %initial guess of parameters
      O = 14;          %Number of coefficients in a vector 
      T = 89;         %Number of vectors in a sequence 
      nex =numel(matfiles);        %Number of sequences 
      M = 3;          %Number of mixtures 
      Q = 3;          %Number of states 

    data = tmpdata;
    prior0 = normalise(rand(Q,1));
    transmat0 = mk_stochastic(rand(Q,Q));
    cov_type = 'ppca';

    if 0
      Sigma0 = repmat(eye(O), [1 1 Q M]);
      % Initialize each mean to a random data point
      indices = randperm(T*nex);
      mu0 = reshape(data(:,indices(1:(Q*M))), [O Q M]);
      mixmat0 = mk_stochastic(rand(Q,M));
    else
      [mu0, Sigma0] = mixgauss_init(Q*M, data, cov_type);
      mu0 = reshape(mu0, [O Q M]);
      Sigma0 = reshape(Sigma0, [O O Q M]);
      mixmat0 = mk_stochastic(rand(Q,M));
    end

    [LL, prior1, transmat1, mu1, Sigma1, mixmat1] = ...
        mhmm_em(data, prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 10);


    loglik = mhmm_logprob(data, prior1, transmat1, mu1, Sigma1, mixmat1);

    gmm.prior = prior1;
    gmm.transmat = transmat1;
    gmm.mu = mu1;
    gmm.sigma = Sigma1;
    gmm.mixmat = mixmat1;
end

function models = trainEachDim(class)
    global M Q O

    tmpdata = getData('train', class);

    for dim = 1 : 14
        fprintf('dim: %d\n', dim);
        data = tmpdata(dim, :, :);
        prior0 = normalise(rand(Q,1));
        transmat0 = mk_stochastic(rand(Q,Q));
        cov_type = 'spherical';

        if 0
          Sigma0 = repmat(eye(O), [1 1 Q M]);
          % Initialize each mean to a random data point
          indices = randperm(T*nex);
          mu0 = reshape(data(:,indices(1:(Q*M))), [O Q M]);
          mixmat0 = mk_stochastic(rand(Q,M));
        else
          [mu0, Sigma0] = mixgauss_init(Q*M, data, cov_type);
          mu0 = reshape(mu0, [O Q M]);
          Sigma0 = reshape(Sigma0, [O O Q M]);
          mixmat0 = mk_stochastic(rand(Q,M));
        end

        [LL, prior1, transmat1, mu1, Sigma1, mixmat1] = ...
            mhmm_em(data, prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 100, 'cov_type', cov_type);

%         loglik = mhmm_logprob(data, prior1, transmat1, mu1, Sigma1, mixmat1);

        gmm.prior = prior1;
        gmm.transmat = transmat1;
        gmm.mu = mu1;
        gmm.sigma = Sigma1;
        gmm.mixmat = mixmat1;
        models(dim) = gmm;
    end
end

function [trainData, trainLabel, testData, testLabel]=weakTrainData()
    [trainData, trainLabel]=reassembleData('train');
    [testData, testLabel]=reassembleData('test');
    save('trainData.mat', 'trainData');
    save('trainLabel.mat', 'trainLabel');
    save('testData.mat', 'testData');
    save('testLabel.mat', 'testLabel');
end

function [data, label] = reassembleData(mode)
    load models;
    data = [];
    label = [];
    for c = 0 : 2
        rawData = getData(mode, c);
        for q = 1 : size(rawData ,3)
            tmpdata = rawData(:,:,q);
            for i = 1 : size(models, 1)
                for j = 1 : size(models, 2)
                    gmm = models(i,j);
                    loglik(i,j,q) = mhmm_logprob(tmpdata(j,:), gmm.prior, gmm.transmat, gmm.mu, gmm.sigma, gmm.mixmat);
                end
            end
        end
        tmp = reshape(loglik, [size(loglik,1)*size(loglik,2) size(loglik, 3)]);
        data = [data, tmp];   
        label = [label, repmat(c, [1, size(tmp, 2)])];
    end

end
