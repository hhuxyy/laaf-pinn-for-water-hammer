Program single_pipe
    use shell32
    implicit none
    real(KIND=8),parameter::pi=3.14159,g=9.806
    real(KIND=8)::Hr=50,L=300,c=1000,D=0.05,f=0.015,tc=0.1,Tmax=6,Em=1.0,CdAg=0.009        !�����еĸ�������
    !real(KIND=8)::Hr=20,L=1160,c=1000,D=0.02,f=0.03,tc=0.1,Tmax=6,Em=1.0,CdAg=0.009        !�����еĸ�������
    integer::N
    real(KIND=8),allocatable::Hp(:),Qp(:)                 !���Դ洢����ʱ�̵�ˮͷ������
    real(KIND=8),allocatable::H(:),Q(:)                  !��ʱ�̵�ˮͷ������ 
    real(KIND=8),allocatable::Qpp(:)             !����ʱ�̵�ˮͷ������
    real(KIND=8),allocatable::REP(:)
    REAL(KIND=8),ALLOCATABLE::UF_R_YN(:,:),UF_R_YP(:,:),UF_R_YPP(:,:)
    REAL(KIND=8)::UF_R_TAU
    real(KIND=8)::Cp,Cm,Bp,Bm                          !�����߲���
    real(KIND=8)::Q0,H0,Cv                             !��ʼ�����뷧�Ŵ���ʼˮͷ��ʧ,��ⷧ���м����
    real(KIND=8)::B,R,A                                !���������迹B���迹ϵ��,�ܵ����A
    !INTEGER,parameter::dx=5
    integer,parameter::uf_choice=1
    real(kind=8)::dx
    real(KIND=8)::dt=0.005                               !�ܵ�ÿ�γ��Ⱥ�˲�����ļ��ʱ��
    real(KIND=8)::t=0,TAU=1                            !��ʼʱ���뷧�ſ���
    integer::ti=0                               !�ڼ�������ʱ��
    real(KIND=8)::f_open                               !�򿪼������ļ��Ĳ���
    integer::i                                 !�ܵ��ڽڵ��ѭ��
    real(KIND=8)::j                                    !ʱ��ѭ��
    integer::k
    integer,parameter::fileHi = 201
    integer,parameter::fileQi = 202
    !����ض�����
    !N = L/dx

    !dx= L/N
    dx = dt*c
    !dt=dx/c
    N=int(L/dx)
    allocate(Hp(1:N+1),Qp(1:N+1))
    allocate(H(1:N+1),Q(1:N+1))
    allocate(Qpp(1:N+1))
    allocate(rep(1:N+1))
    ALLOCATE(UF_R_YN(1:N+1,1:17))
    ALLOCATE(UF_R_YP(1:N+1,1:17))
    ALLOCATE(UF_R_YPP(1:N+1,1:17))
    A=pi/4*D*D
    B=c/A/g
    R=f*dx/2/g/D/A/A
    
    
    !��ʼ����
    H(1)=Hr
   ! Q0=sqrt(Hr*2*g*CdAg**2/(R*N*2*g*CdAg**2+1))   !��ʼ����
    Q0 = 0.412*A
    !Q0=0.01
    Q(1)=Q0
    Do i=1,N
        H(i+1)=H(i)-R*Q0**2
        Q(i+1)=Q0
    end do
    !H0=Q0**2/2/g/CdAg**2        !���Ŵ���ʼˮͷ��ʧ
    H0=H(N+1)
   
   !����ʼ��������CSV�����
    open(201,file='Hi.CSV')     
    open(202,file = 'Qi.CSV')
    write(201,"(A,',',A,',',A,',',A,',',A,',',A)")'ʱ��','H1','H2','H3','HT1','HT2'
    write(201,"(F20.5,',',F20.5,',',F20.5,',',F20.5,',',F20.5,',',F20.5)")t,H(17),H(33),H(49),H(11),H(41)
    write(202,"(A,',',A,',',A,',',A,',',A,',',A)")'ʱ��','Q1','Q2','Q3','QT1','QT2'
    write(202,"(F20.5,',',F20.5,',',F20.5,',',F20.5,',',F20.5,',',F20.5)")t,Q(17),Q(33),Q(49),Q(11),Q(41)
    
    
    t=t+dt
    ti=ti+1
    !k =0
    Do j=t,Tmax,dt
        
        if (ti==1) then
        Do i=1,N+1
            QPP(I)=Q(I)
            rep(i)=abs(q(i))*D/A*10**6
        ENDDO
        endif 
        !�ܵ��м�㴦��ˮͷ��������
        Do i=2,N
            Bp=B+R*abs(Q(i-1))
            if (uf_choice==1)then
            
            K = I-1
            CALL SUB_UF(N,TI,I,K,D,C,A,DT,Q,QPP,REP,UF_R_YN,UF_R_YP,UF_R_YPP,UF_R_TAU)
            CP = H(I-1)+B*Q(I-1)-4*dx/(1000*G*D)*UF_R_TAU
            else
                Cp=H(i-1)+B*Q(i-1)
            endif 
            
            Bm=B+R*abs(Q(i+1))
            if (uf_choice==1)then
            K = I+1
            CALL SUB_UF(N,TI,I,K,D,C,A,DT,Q,QPP,REP,UF_R_YN,UF_R_YP,UF_R_YPP,UF_R_TAU)
            CM = H(I+1)-B*Q(I+1)+4*dx/(1000*G*D)*UF_R_TAU
            else
            Cm=H(i+1)-B*Q(i+1)
            endif
            
            HP(i)=(Cp*Bm+Cm*Bp)/(Bp+Bm)
            QP(i)=(Cp-Cm)/(Bp+Bm)
        end do
        
        !���α߽�����
        HP(1)=Hr
        Bm=B+R*abs(Q(2))
        !Cm=H(2)-B*Q(2)
        if (uf_choice==1)then
        I=1
        K=2
        CALL SUB_UF(N,TI,I,K,D,C,A,DT,Q,QPP,REP,UF_R_YN,UF_R_YP,UF_R_YPP,UF_R_TAU)
        CM = H(I+1)-B*Q(I+1)+4*dx/(1000*G*D)*UF_R_TAU
        else
        Cm=H(2)-B*Q(2)
        endif
        Qp(1)=(Hp(1)-Cm)/Bm
        
        !���α߽�����
        if (j<=Tc) then
            TAU=(1-j/Tc)**Em
        else
            TAU=0
        end if
        Cv=(Q0*TAU)**2/2/H0
        Bp=B+R*abs(Q(N))
   
        if (uf_choice==1)then
        I=N+1
        K=N
        CALL SUB_UF(N,TI,I,K,D,C,A,DT,Q,QPP,REP,UF_R_YN,UF_R_YP,UF_R_YPP,UF_R_TAU)
        CP = H(I-1)+B*Q(I-1)-4*dx/(1000*G*D)*UF_R_TAU
        else
        Cp=H(N)+B*Q(N)
        endif
        
        Qp(N+1)=-Bp*Cv+sqrt(Bp**2*Cv**2+2*Cv*Cp)
        Hp(N+1)=Cp-Bp*Qp(N+1)
        
        Do i=1,N+1
            QPP(I)=Q(I)
            H(i)=Hp(i)
            Q(i)=Qp(i)
        end do
        Do i=1,N+1
            rep(i)=abs(q(i))*D/A*10**6
        ENDDO
        !k = k+1
        !if (mod(k,5)==0) then 
        write(201,"(F20.5,',',F20.5,',',F20.5,',',F20.5,',',F20.5,',',F20.5)")J,H(17),H(33),H(49),H(11),H(41)
        write(202,"(F20.5,',',F20.5,',',F20.5,',',F20.5,',',F20.5,',',F20.5)")J,Q(17),Q(33),Q(49),Q(11),Q(41)
        !end if 
        ti=ti+1
        
    end do
    close(1)
    f_open=shellexecute(0,"open","Hi.CSV",null,null,1)
end