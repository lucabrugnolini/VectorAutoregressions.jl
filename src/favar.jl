using DataFrames, Plots, MultivariateStats, GLM

# The code is taken from the original 
# Bernanke Boivin and Eliasz paper 
# FAVAR Bayesian structural factor augmented VAR
# -------------------------------------------------------------------------
# The model is:
# 
#   FACTOR (OBSERVATION) EQUATION:
#     _      _           _      _
#    |  X(t)  |         |  F(t)  |
#    |        |  = L  x |        | + e(t) 
#    |_ Y(t) _|         |_ Y(t) _|
# 
# 
#   VAR (STATE) EQUATION:
#     _      _            _        _
#    |  F(t)  |          |  F(t-1)  |
#    |        |  = PHI x |          |  +  u(t) 
#    |_ Y(t) _|          |_ Y(t-1) _|
# 
# 
#  with u(t)~N(0,S_F), and e(t)~N(0,Sigma), with Sigma = diag{(sigma_1)^2 ,
#  ...., (sigma_N)^2}.
# -------------------------------------------------------------------------
#-----------------------------LOAD DATA------------------------------------
# Load quarterly data
# load data used to extract factors
xdata = readtable("/home/lbrugnol/Downloads/FAVAR/xdata.csv", header = true)
# load data on inflation, unemployment and interest rate 
ydata = readtable("/home/lbrugnol/Downloads/FAVAR/ydata.csv", header = false)
# load transformation codes (see file transx.m)
# load the slow or fast moving variables codes (see Bernanke, Boivin and
# Eliasz, 2005, QJE)
tcode = readtable("/home/lbrugnol/Downloads/FAVAR/tcode.csv", header=false)
slowcode = readtable("/home/lbrugnol/Downloads/FAVAR/slowcode.csv", header = false)
# load the file with the dates of the data (quarters)
yearlab = readtable("/home/lbrugnol/Downloads/FAVAR/yearlab.csv", header = false)
# load the file with the names of the variables. 
namesX = readtable("/home/lbrugnol/Downloads/FAVAR/namesX.csv", header = false)

# namesX.mat contains only the short codes of the data. For a complete
# description see the attached excel file.
stddata = [std(convert(Array,xdata),1) ones(1,3)]

const dRule = Dict{String,Integer}("none" => 1, "first_diff" => 2, "log_yoy"=> 3,
"log" => 4, "log_first_diff" => 5, "yoy" => 6)
const sFile_sd = "/home/lbrugnol/Downloads/FAVAR/xdata.csv"
const bHeader_sd = true
const sFile_tc = "/home/lbrugnol/Downloads/FAVAR/tcode.csv"
const bHeader_tc = false
# const sStart = "1999-01-31"

# Call starting dataset, transformation and create final dataset
sd = StartData(sFile_sd,bHeader_sd)
tc = TransRule(sFile_tc,bHeader_tc,dRule,sd)
fd = FinalData(sd,tc)

# Correct size after stationarity transformation
xdata = convert(Array,fd.dfData[3:end,:]) 
ydata = convert(Array,ydata[3:end,:]) 
yearlab = yearlab[3:end,:] 

# Demean data (no intercept is estimated as in BBE (2005))
t1 = size(xdata,1)     # time series observations of xdata
t2 = size(ydata,1)     # time series dimension of ydata
stdffr = std(ydata[:,3],1)       # standard deviation of the Fed Funds Rate
xdata = xdata .- mean(xdata,1) 
ydata = ydata .- mean(ydata,1) 

# Define X and Y matrices
X = xdata    # X contains the 'xdata' which are used to extract factors.
Y = ydata  # Y contains inflation, unemployment and interest rate
# NamesXY has the short codes of X and Y
namesXY = [fd.sd.vNames; "Inflation"; "Unemployment"; "Fed_funds"]

# Number of observations and dimension of X and Y
T,M=size(Y) # T time series observations
N=size(X,2) # N series from which we extract factors
#----------------------------PRELIMINARIES---------------------------------
# Set some Gibbs - related preliminaries
const nrep = 15000  # Number of replications
const nburn = 5000   # Number of burn-in-draws
const nthin = 1   # Consider every thin-th draw (thin value)
const it_print = 100  #Print in the screen every "it_print"-th iteration
const constant = 1  # Set 0:no constant on the FAVAR equation, 1:include constant

# Number of factors & lags:
const K = 2               # Number of Factors
const p = K+M             # p is the dimensionality of [Factors, Y]
const plag = 2            # plag is number of lags in the VAR part
# ==============================| FACTOR EQUATION |========================
# Extract principal components from X (to be used as starting values for the
# MCMC estimation of the factors)
X_st = X./std(X,1)# First standardize data to extract PC
[F0,Lf]=extract(X_st,K)  # F0 are the factors, Lf are the loadings

##### UNDERSTAND WHY DIFFERENT FACTORS WRT MATLAB!!!!!!

function pca(data::Array{Float64}; n::Int64 = 1)
    # Need MultivariateStats
    mX = data.-mean(data,1)
    M = fit(PCA, mX, method = :cov)
    return projection(M)[:,1:n]
  end

F0 = pca(X_st,n = K)

#----If using PC estimates of the factors:
# Now rotate the factor space as in Bernanke, Boivin and Eliasz (2005)
slowindex = find(convert(Array,slowcode).==1)
xslow = X[:,slowindex]
Fslow0 = pca(xslow,n = K)

Ffast0 = Y[:,end]
t1,k1=size(Ffast0[:,:])
ff = ([ones(t1,1) Ffast0 Fslow0]'*[ones(t1,1) Ffast0 Fslow0])
fF = [ones(t1,1) Ffast0 Fslow0]'*F0
b = ff\fF # original paper ols svd
Fr0 = F0 - Ffast0*b[2:k1+1,:]

X=X_st
Y=Y./std(Y)

# #----If using MCMC estimates of the factors:
# # This part is explained in the PhD thesis of Eliasz (2005). Essentially,
# # we rotate the factors in a certain way which gives a loadings matrix L,
# # with the upper KxK block being the identity matrix. This structure of L
# # is the same structure we will have when using MCMC to estimate the
# # factors (because of the identification restrictions imposed on L).
# # regress X on F0 and Y, obtain loadings
# Lfy=olssvd(X(:,K+1:N),[F0 Y])'     # upper KxM block of Ly set to zero
# Lf=[Lf(1:K,:);Lfy(:,1:K)]
# Ly=[zeros(K,M);Lfy(:,K+1:K+M)]
# # transform factors and loadings for LE normalization
# [ql,rl]=qr(Lf')
# Lf=rl  # do not transpose yet, is upper triangular
# F0=F0*ql
# # need identity in the first K columns of Lf, call them A for now
# A=Lf(:,1:K)
# Lf=[eye(K),inv(A)*Lf(:,(K+1):N)]'
# Fr0=F0*A

# Put it all in state-space representation, write obs equ as XY=FY*L+e
XY=[X Y]    # the data X used to extract factors, plus Y (infl,unemp and interest)
FY=[Fr0 Y]  # the extracted factors, plus Y (infl,unemp and interest)

# Obtain L (the loadings matrix)
#----If using PC estimates of the factors:
L = (FY'*FY)\FY'*XY # original paper ols svd decom

# #----If using MCMC estimates of the factors:
# L=[Lf Ly;zeros(M,K),eye(M)] 

# Obtain SIGMA (the error varaince in the factor equation)
e = XY - FY*L
SIGMA = e'*e./T
SIGMA = diagm(diag(SIGMA))

# ================================| VAR EQUATION |=========================
# Generate lagged FY matrix.
function mlag2(mX,p)
    T,N = size(mX)
    mXlag = zeros(T,N*p)
    for ii in 1:p
        mXlag[p+1:T,(N*(ii-1)+1):N*ii] = mX[p+1-ii:T-ii,1:N]
    end
    return mXlag
end

FY_lag = mlag2(FY,plag)



FY_lag = FY_lag[plag+1:T,:]
FY_temp = FY[plag+1:T,:]

# Get an initial estimate of PHI (FAVAR autoregressive coefficients) and
# S_F (FAVAR covariance matrix). These values will be used as initial
# values for the Gibbs sampler (but you can use 'zeros' instead, to declare
# 'ignorance' about the initial values 
PHI = inv(FY_lag'*FY_lag)*(FY_lag'*FY_temp)
SSE = (FY_temp - FY_lag*PHI)'*(FY_temp - FY_lag*PHI)
S_F = SSE./(T-p)

# If plag is not 1, we need to write the state equation as a VAR(1) model,
# in order to be able to estimate the latent factors using the Kalman
# filter. In this case, PHI and S_F are the parameter matrices of the VAR(p)
# model, while PHI_mat and S_F_mat are the matrices of the VAR(1)
# transformed model (see the time series book of Luthkephol (2005), for
# example). If using PC estimates this is not necessary, but we will need
# the VAR(1) form anyway for the impulse responses, so do not comment this
# part.
S_F_mat = [S_F zeros(p,p*(plag-1));zeros(p*(plag-1),p*plag)]
PHI_mat = [PHI' ; eye(p*(plag-1)) zeros(p*(plag-1),p)]

#-------------------------------- PRIORS ----------------------------------
# #========= INITIAL CONDITION ON LATENT FACTORS:
# # F(0) ~ N(F_0_prmean, F_0_prvar)
# F_0_prmean = zeros(p*plag,1);
# F_0_prvar = 4*eye(p*plag)
# 
# # variable indexnM indexes which draws of FY refer only to the factors F 
# # (and not Y, since Y is assumed to be observed and known). See also
# # function kfgibbsnv.m
# indexnM = [ones(K,plag);zeros(M,plag)]
# indexnM = find(indexnM==1)

# PRIORS ON FACTOR EQUATION:
# Prior on loadings L_i ~ N(0, I), where i = 1,...,N
Li_prvar = 4*eye(p)
alpha = 0.01
beta = 0.01


# Prior on covariance SIGMA_i ~ iG(a, b), where i = 1,...,N 
alpha = 0.01
beta = 0.01

# PRIORS ON VAR EQUATION:


# IMPULSE RESPONSES:
# Note that impulse response and related stuff involves a lot of storage
# and, hence, put istore=0 if you do not want them
istore = 1
if istore == 1
    # Impulse response horizon
    nhor = 21
    shock_init = diagm([zeros(p-1); 1./stdffr]) # in terms of standard deviation, identification is recursive
    imp = zeros(nrep,N+M,nhor)
    bigj = zeros(p,p*plag)
    bigj[1:p,1:p] = eye(p)
end
#-------------------------- END OF PRELIMINARIES --------------------------

# START SAMPLING 
println("Number of iterations")

for irep = 1:nrep + nburn    #  GIBBS iterations starts here
    # Print iterations
    # if mod(irep,it_print) == 0
    #     disp(irep); toc
    # end
    
    #FACTOR (MEASUREMENT) EQUATION: 
    # -----------------------------------------------------------------------------------------
    #   STEP I: Sample latent factors F_t using Carter and Kohn (1994) or
    #   use PC estimates
    # ----------------------------------------------------------------------------------------- 
    
    # #----If using MCMC estimates of the factors:
    # FY = kfgibbsnv(XY,F_0_prmean,F_0_prvar,L,SIGMA,PHI_mat,S_F_mat,M,indexnM)
    #     
    # # Make sure the factors have mean zero
    # FY = FY - repmat(mean(FY),T,1)

    # Commenting the above, means that we are using the PC estimate of FY.
    
    # -----------------------------------------------------------------------------------------
    #   STEP II: Sample L and SIGMA
    # ----------------------------------------------------------------------------------------- 
    # Since the covariance matrix of the error (SIGMA) in this equation is
    # diagonal, we can estimate the parameters equation-by-equation
    for i = 1:N
        # Sample L from a Normal distribution. The upper KxK block of L is 
        # the identity matrix, so we sample the rest N-K rows only:
        if i > K
            Li_postvar = inv(inv(Li_prvar) + inv(SIGMA[i,i])*FY'*FY)
            Li_postmean = Li_postvar*(inv(SIGMA[i,i])*FY'*X[:,i])
            Lidraw = Li_postmean' + randn(1,p)*chol(Hermitian(Li_postvar))
            L'[i,1:p] = Lidraw
        end
        
        ed = X[:,i] - FY*L'[i,:]

        # Sample SIGMA(i,i) from iGamma
		S_1 = alpha/2 + T/2
		S_2 = beta/2 + ((X[:,i] - FY*L'[i,:])'*(X[:,i] - FY*L'[i,:]))/2       
        Sidraw = inv(rand(Gamma(S_1,1/S_2)))
        SIGMA[i,i] = Sidraw
    end  
    
    #VAR (STATE) EQUATION:
    # Generate lagged FY and then correct sizes (we loose observations
    # because of taking lags, as with the simple VAR model)
    FY_lag = mlag2(FY,plag)
    FY_lag = FY_lag[plag+1:T,:]

    FY_temp = FY[plag+1:T,:]
    # Now we have everything we need to estimate PHI and S_F, as we did
    # with the simple BVAR models. You can use any of the priors we used in
    # the BVAR models, like the Diffuse, the Normal-Wishart, the Independent
    # Normal-Wishart, or the Minnesota (Littermann) prior. Here I use the
    # Diffuse prior, based on the OLS quantities (remember, at each Gibbs
    # iteration we treat the dependent variables (FY_temp) and the lags of
    # the dependent variables (FY_lag) as observed data.
        
    # -----------------------------------------------------------------------------------------
    #   STEP I: Sample autoregressive coefficients PHI
    # -----------------------------------------------------------------------------------------
    
    PHI_OLS = inv(FY_lag'*FY_lag)*(FY_lag'*FY_temp)
    phi_OLS = PHI_OLS(:)
    
    V_post = kron(S_F,inv(FY_lag'*FY_lag))
    phi_vec = phi_OLS + chol(V_post)'*randn(p*p*plag,1)
    PHI = reshape(phi_vec,p*plag,p)'

    # Now create draw of PHI_mat, the parameters in the VAR(1) transformation
    PHI_mat(1:p,:) = PHI
    
#     # truncate to ensure stationarity
#     while max(abs(eig(PHI_mat)))>0.999
#         phi_vec = phi_OLS + chol(V_post)'*randn(p*p*plag,1)
#         PHI = reshape(phi_vec,p*plag,p)'   
#         PHI_mat(1:p,:) = PHI
#     end
    # -----------------------------------------------------------------------------------------
    #   STEP II: Sample covariance matrix S_F
    # -----------------------------------------------------------------------------------------
    SSE = (FY_temp - FY_lag*PHI_OLS)'*(FY_temp - FY_lag*PHI_OLS)
    S_F = inv(wish(inv(SSE),T-p))
    
    # Now create draw of S_F_mat, the parameter in the VAR(1) transformation
    S_F_mat(1:p,1:p) = S_F
    
    
    #--------------------------SAVE AFTER-BURN-IN DRAWS AND IMPULSE RESPONSES -----------------
    if irep > nburn && mod((irep-nburn),nthin)==0
        
        # Save here draws of the parameters to get means and variances
  
        
        #----------------- Impulse response analysis:
        if istore==1
            # Note that Htsd contains the
            # structural error cov matrix
            # Set up things in VAR(1) format as in Lutkepohl page 11      
            biga = PHI_mat
            
            # st dev matrix for structural VAR
            shock = chol(S_F)'
            d = diag(diag(shock))
            shock = inv(d)*shock
            #shock = Hsd2*shock_init
            #shock=eye(p)
            
            #now get impulse responses for 1 through nhor future periods@
            impresp = zeros(p,p*nhor)
            impresp(1:p,1:p) = shock
            bigai = biga
            for j = 1:nhor-1
                impresp(:,j*p+1:(j+1)*p) = bigj*bigai*bigj'*shock
                bigai = bigai*biga
            end
            
            imp_m = zeros(p,nhor)
            jj=0
            for ij = 1:nhor
                jj = jj + p
                imp_m(:,ij) = impresp(:,jj)
            end
            imp(irep-nburn,:,:) = L*imp_m
       end #END the impulse response calculation section         
    end # END saving after burn-in results 
end #END main Gibbs loop (for irep = 1:nrep+nburn)
#=============================GIBBS SAMPLER ENDS HERE==================================


if istore == 1
    scale=repmat(stddata',[1 3 nhor])
    scale=permute(scale,[2 1 3])
    
    imp = permute(imp,[2 3 1])
    
    for i=1:N
        if tcode(i)==4
            imp(i,:,:) = exp(imp(i,:,:))-ones(1,nhor,nrep/nthin)
        elseif tcode(i)==5
            imp(i,:,:)=exp(cumsum(imp(i,:,:),2))-ones(1,nhor,nrep/nthin)
        end
    end
    imp = permute(imp,[3 1 2])
    
    # Set quantiles from the posterior density of the impulse responses
    qus = [.10, .5, .9]
    #if using PC estimates of the factors, use this
    impXY=squeeze(quantile(imp,qus))
#     #if using MCMC estimates of the factors, use this
#     impXY=squeeze(quantile(imp,qus))./scale    
    
    #====================PLOTS    
    #---Plot I: impulse responses of inflation, unemployment, interest
    figure       
    set(0,'DefaultAxesColorOrder',[0 0 0],...
        'DefaultAxesLineStyleOrder','--|-|--')
    subplot(3,1,1)
    plot(1:nhor,squeeze(impXY(:,end-2,:)))
    hold
    plot(zeros(1,nhor),'-')
    title('Impulse response of inflation')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)
    subplot(3,1,2)
    plot(1:nhor,squeeze(impXY(:,end-1,:)))
    hold
    plot(zeros(1,nhor),'-')
    title('Impulse response of unemployment')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,1,3)
    plot(1:nhor,squeeze(impXY(:,end,:)))
    hold
    plot(zeros(1,nhor),'-')
    title('Impulse response of interest rate')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)
    #-------------------------
    
    #---Plot II: impulse responses of other variables
    figure       
    set(0,'DefaultAxesColorOrder',[0 0 0],...
        'DefaultAxesLineStyleOrder','--|-|--')
    # I will plot only 12 out of the 115 variables here
    var_numbers = [2  9 10 28 42 46 77 91 92 108 109 111]
    # These variables have the following short codes:
    var_names = namesXY(var_numbers)
    
    for i=1:12
        subplot(4,3,i)   
        plot(1:nhor,squeeze(impXY(:,var_numbers(i),:)))
        hold
        plot(zeros(1,nhor),'-')
        title(['Impulse response of ' var_names(i)])   
        xlim([1 nhor])
        set(gca,'XTick',0:3:nhor)   
    end
    #-------------------------
    
end

clc
toc # Stop timer and print total time