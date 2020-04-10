using SparseArrays
using LinearAlgebra
using DelimitedFiles
function eye(x::Any)
    res = Matrix{Float64}(I, x,x);
    return res
end

mutable struct mp_struct
	n::Int64
	max::Int64
	grid::Vector{Float64}
	p::Vector{Float64}
	RC::Float64
	c::Float64
	beta::Float64
end
function mp_mod(;#
	n=175,# % Number of gridpoints
	max=450,	# % Max of mileage
	grid = [0:1:n-1;], # 	% Grid over mileage
	#% Structural parameters
	p=[0.0937, 0.4475, 0.4459, 0.0127], #  	% Transition probabiliuties
	RC=11.7257,  #% Replacement cost
	c=2.45569,	#% Cost parameter
	beta=0.9999
	)
	return mp_struct(n,max,grid,p,RC,c,beta)
end
mp = mp_mod()
function bellman(ev::Vector{Float64}, P::SparseMatrixCSC, c::Vector{Float64}, mp::mp_struct)
	#=
	% ZURCHER.BELMANN:     Procedure to compute bellman equation
	%
	% Inputs
	%  ev0      mp.n x 1 matrix of expected values given initial guess on value function
	%
	% Outputs:
	%  ev1      mp.n x 1 matrix of expected values given initial guess of ev
	%  pk       mp.n x 1 matrix of choice probabilites (Probability of keep)
	=#
	VK=-c.+mp.beta .*ev;               #% Value off keep
	VR=-mp.RC-c[1]+mp.beta*ev[1];   #% Value of replacing
	maxV=max.(VK, VR); #elementwise max

	#% Need to recentered by Bellman by subtracting max(VK, VR)
	ev1=P*(maxV .+ log.(exp.(VK.-maxV)  .+  exp.(VR.-maxV)));

	#% If requested, also compute choice probability from ev (initial input)
	pk=1.0 ./(1.0 .+exp.(VR.-VK));
	#% compute Frechet derivative
	dGamma_dev=dbellman(pk, P, mp);
	return ev1, pk, dGamma_dev
end

function dbellman(pk,  P::SparseMatrixCSC, mp::mp_struct)
	#% ZURCHER.DBELMANN:     Procedure to compute Frechet derivative of Bellman operator
	#dGamma_dEv=sparse(mp.n,mp.n);
	dGamma_dEv=sparse(Matrix(0.0I, mp.n, mp.n)) # initialize all zero matrix
	dGamma_dEv=mp.beta.* (P.*pk);   # %
	dGamma_dEv[:,1]=dGamma_dEv[:,1].+mp.beta.*P*(1 .-pk);  # % Add additional term for derivative wrt Ev(1), since Ev(1) enter logsum for all states

	#% Alternative way to express this.
	#% dGamma_dEv(:,1)= mp.beta*(P(:,1)*pk(1) + P*(1-pk));
	#% dGamma_dEv(:,2:mp.n)=mp.beta*bsxfun(@times, P(:,2:mp.n), pk(2:mp.n)');    %
	return dGamma_dEv
end

function statetransition(p::Vector{Float64}, n::Int64)
	p=[p; (1-sum(p))];
	P=0;
	for i=0:length(p)-1;
		#P=P+sparse(1:n-i,1+i:n,ones(1,n-i)*p(i+1), n,n);
		sparse_p = vcat(sparse(1:n-i,1+i:n,vec(ones(1,n-i)*p[i+1])),zeros(i,n)) # adjust dimension by vcat
		P=P.+sparse_p;
		P[n-i,n]=1-sum(p[1:i]);
	end
	#P=sparse(P);
	P=sparse(P);
	return P
end
#% Transition matrix for mileage
P0 = statetransition(mp.p, mp.n);

function bellman(ev::Vector{Float64})
	ev1, pk, dGamma_dev = bellman(ev, P0, cost0, mp);
	return ev1, pk, dGamma_dev
end

#% Initial guess on ev
ev0=[0.0]; # must be Vector
#% Cost function
cost0=0.001*mp.c*mp.grid;
V0 = ev0
V1, P, dV=bellman(V0); #% also return value and policy function
m = length(ev0)
#F=speye(m) - dV; #% using dV from last call to bellman
F_julia=sparse(eye(m)) .- dV
println("Caution!! Following Inverse Matrix produces different output from Maltab!")
V=V0.-F_julia\(V0.-V1); #% NK-iteration x = A\B do A*x = B.
V=V0.-inv(Matrix(F_julia))*(V0.-V1); # avoid sparse matrix issue



#---------------------------------
# Following data comes from Matlab
#----------------------------------
F_matlab = readdlm("F.txt", ',',Float64) # corresponds to F
invF_matlab = readdlm("invF.txt", ',',Float64) # corresponds to inv(F)
F_V0_V1_matlab = readdlm("F_V0_V1.txt", ',',Float64) # corresponds to F\(V0.-V1)

# the following differences should be zero
@show maximum(F_matlab.-F_julia) # zero correctly or approximately
@show maximum(F_matlab.-round.(F_julia,digits=5)) # rounding because matlab rounds the number
@show maximum(invF_matlab.-inv(Matrix(F_julia)))
@show maximum(invF_matlab.-inv(F_matlab))
@show maximum(F_V0_V1_matlab.-(F_julia\(V0.-V1)))

# check the identity propety for F_julia
F_julia*inv(Matrix(F_julia))
@show diag(F_julia*inv(Matrix(F_julia)))
inv(Matrix(F))*F
@show diag(inv(Matrix(F_julia))*F_julia)
F_julia*invF_matlab
@show diag(F_julia*invF_matlab)
invF_matlab*F_julia
@show diag(invF_matlab*F_julia)

# check the identity propety for F_matlab
F_matlab*inv(Matrix(F_matlab))
@show diag(F_matlab*inv(Matrix(F_matlab)))
inv(Matrix(F))*F
@show diag(inv(Matrix(F_matlab))*F_matlab)
F_matlab*invF_matlab
@show diag(F_matlab*invF_matlab)
invF_matlab*F_matlab
@show diag(invF_matlab*F_matlab)
