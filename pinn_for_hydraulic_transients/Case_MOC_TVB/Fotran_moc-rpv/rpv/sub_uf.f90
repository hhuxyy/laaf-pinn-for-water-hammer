
SUBROUTINE SUB_UF(PIPR_I_NS                            ,&!�ܶηֶ���                 ����         (    /    )    //�������
                  PIPR_I_T                             ,&!����ʱ������                      ����         (    s    )    //�������
                  PIPR_I_J                             ,&!�ܵ��ڵ�                          ����         (    /    )    //�������
                  PIPR_I_K                             ,&!PIPR_I_J��(��)�ڵ�                ����         (    /    )    //�������
                  PIPR_R_D                             ,&!�ܵ�ֱ��                          ʵ��         (    m    )    //�������
                  PIPR_R_A                             ,&!�ܵ�����                          ʵ��         (   m/s   )    //�������
                  PIPR_R_AS                            ,&!�ܵ����                          ʵ��         (   m2    )    //�������
                  PIPR_R_DT                            ,&!ʱ�䲽��                          ʵ��         (    s    )    //�������
                  PIPR_R_QP                            ,&!ǰһʱ������                      ʵ��         (   m3/s  )    //�������
                  PIPR_R_QPP                           ,&!ǰǰʱ������                      ʵ��         (   m3/s  )    //�������
                  PIPR_R_REP                           ,&!��һʱ����ŵ��                    ʵ��         (    /    )    //�������
                  UF_R_YN                              ,&!������ֱ�ʱ��Y                   ʵ��         (    /    )    //�������/�������
                  UF_R_YP                              ,&!�������ǰʱ��Y                   ʵ��         (    /    )    //�������/�������
                  UF_R_YPP                             ,&!�������ǰǰʱ��Y                 ʵ��         (    /    )    //�������/�������
                  UF_R_TAU)                              !���������                        ʵ��         (    /    )    //�������

    IMPLICIT NONE
    
    INTEGER:: PIPR_I_NS                               !���м���ڵ�����                  ����         (    /    )    //�������
    INTEGER::PIPR_I_T                                 !����ʱ������                      ����         (    s    )    //�������
    INTEGER::PIPR_I_J                                 !�ܵ��ڵ�                          ����         (    /    )    //�������
    INTEGER::PIPR_I_K                                 !PIPR_I_J��(��)�ڵ�                ����         (    /    )    //�������
    REAL(KIND=8)::PIPR_R_D                                    !�ܵ�ֱ��                          ʵ��         (    m    )    //�������
    REAL(KIND=8)::PIPR_R_A                                    !�ܵ�����                          ʵ��         (   m/s   )    //�������
    REAL(KIND=8)::PIPR_R_AS                                   !�ܵ����                          ʵ��         (   m2    )    //�������
    REAL(KIND=8)::PIPR_R_DT                                   !ʱ�䲽��                          ʵ��         (    s    )    //�������
    REAL(KIND=8)::PIPR_R_QP(1:PIPR_I_NS+1)                      !ǰһʱ������                      ʵ��         (   m3/s  )    //�������
    REAL(KIND=8)::PIPR_R_QPP(1:PIPR_I_NS+1)                     !ǰǰʱ������                      ʵ��         (   m3/s  )    //�������
    REAL(KIND=8)::PIPR_R_REP(1:PIPR_I_NS+1)                     !��һʱ����ŵ��                    ʵ��         (    /    )    //������� 
                                                                                   
    REAL(KIND=8)::MIAB_R_K                                    !MIABģ��ϵ��                      ʵ��         (    /    )    //�м����
    REAL(KIND=8)::PIPR_R_NU                                   !�˶�ճ��                          ʵ��         (    /    )    //�м����
    REAL(KIND=8)::PIPR_R_KS                                   !�ܵ��ֲڶ�                        ʵ��         (    /    )    //�м����
    REAL(KIND=8)::TVB_R_N(1:17)                               !TVBģ��ϵ��                       ʵ��         (    /    )    //�м����
    REAL(KIND=8)::TVB_R_M(1:17)                               !TVBģ��ϵ��                       ʵ��         (    /    )    //�м����  
    REAL(KIND=8)::UF_R_TAU0                                   !��ʼ���������                    ʵ��         (    /    )    //�м����
    REAL(KIND=8)::TVB_R_ASTAR                                 !TVB����ASTAR                      ʵ��         (    /    )    //�м����
    REAL(KIND=8)::TVB_R_BSTAR                                 !TVB����BSTAR                      ʵ��         (    /    )    //�м����
    INTEGER::TVB_I_NUM                                        !ѭ������                          ����         (    /    )    //�м����
    INTEGER::I                                                !ѭ������                          ����         (    /    )    //�м����
    
    REAL(KIND=8)::UF_R_YN(1:PIPR_I_NS+1,1:17)                   !������ֱ�ʱ��Y                   ʵ��         (    /    )    //�������/�������
    REAL(KIND=8)::UF_R_YP(1:PIPR_I_NS+1,1:17)                   !�������ǰʱ��Y                   ʵ��         (    /    )    //�������/�������
    REAL(KIND=8)::UF_R_YPP(1:PIPR_I_NS+1,1:17)                  !�������ǰǰʱ��Y                 ʵ��         (    /    )    //�������/�������
    REAL(KIND=8)::UF_R_TAU                                    !���������                        ʵ��         (    /    )    //�������
     
    
 
        PIPR_R_NU=1.0/10**6
        PIPR_R_KS=0   
        UF_R_TAU0=(4*PIPR_R_NU/PIPR_R_D**2)*PIPR_R_DT
        
        IF(PIPR_R_REP(PIPR_I_K)<=2320) THEN
            TVB_I_NUM=9
            
            TVB_R_N(1)=26.3744
            TVB_R_N(2)=10**2
            TVB_R_N(3)=10**2.5
            TVB_R_N(4)=10**3
            TVB_R_N(5)=10**4
            TVB_R_N(6)=10**5
            TVB_R_N(7)=10**6
            TVB_R_N(8)=10**7
            TVB_R_N(9)=10**8
        
            TVB_R_M(1)=1
            TVB_R_M(2)=2.1830
            TVB_R_M(3)=2.7130
            TVB_R_M(4)=7.5455
            TVB_R_M(5)=39.0066
            TVB_R_M(6)=106.8075
            TVB_R_M(7)=359.0846
            TVB_R_M(8)=1107.9295
            TVB_R_M(9)=3540.6830
            
        ELSE 
            TVB_I_NUM=17
            
            TVB_R_N(1)=10**1
            TVB_R_N(2)=10**1.5
            TVB_R_N(3)=10**2
            TVB_R_N(4)=10**2.5
            TVB_R_N(5)=10**3
            TVB_R_N(6)=10**3.5
            TVB_R_N(7)=10**4
            TVB_R_N(8)=10**4.5
	    	TVB_R_N(9)=10**5
            TVB_R_N(10)=10**5.5
            TVB_R_N(11)=10**6
            TVB_R_N(12)=10**6.5
            TVB_R_N(13)=10**7
            TVB_R_N(14)=10**7.5
            TVB_R_N(15)=10**8
            TVB_R_N(16)=10**8.5
	    	TVB_R_N(17)=10**9
  
            TVB_R_M(1)=9.06	
            TVB_R_M(2)=-4.05	
            TVB_R_M(3)=12
            TVB_R_M(4)=8.05
            TVB_R_M(5)=22.7
            TVB_R_M(6)=35.2
            TVB_R_M(7)=65.9
            TVB_R_M(8)=115
            TVB_R_M(9)=206
            TVB_R_M(10)=365
            TVB_R_M(11)=651
            TVB_R_M(12)=1150
            TVB_R_M(13)=2060
            TVB_R_M(14)=3630
            TVB_R_M(15)=6640
            TVB_R_M(16)=10700
            TVB_R_M(17)=26200
            
            IF(PIPR_R_KS==0) THEN
                TVB_R_ASTAR=0.5/SQRT(3.1415926)
                IF(PIPR_R_REP(PIPR_I_K)<10**8 .AND. PIPR_R_REP(PIPR_I_K)>2000) THEN
                    TVB_R_BSTAR=PIPR_R_REP(PIPR_I_K)**LOG10(15.29/PIPR_R_REP(PIPR_I_K)**0.0567)/12.86   !2003
                ELSEIF(PIPR_R_REP(PIPR_I_K)>=2320) THEN
                    TVB_R_BSTAR=0.135*PIPR_R_REP(PIPR_I_K)**LOG10(14.3/PIPR_R_REP(PIPR_I_K)**0.05)      !1995
                ELSE
                    TVB_R_BSTAR=210
                ENDIF
            ELSE 
                TVB_R_ASTAR=0.0103*SQRT(PIPR_R_REP(PIPR_I_K))*(PIPR_R_KS/PIPR_R_D)**0.39                  !2004
                TVB_R_BSTAR=0.352*PIPR_R_REP(PIPR_I_K)*(PIPR_R_KS/PIPR_R_D)**0.41
            ENDIF 
    
            DO I=1,TVB_I_NUM
                TVB_R_N(I)=TVB_R_N(I)+TVB_R_BSTAR
                TVB_R_M(I)=TVB_R_M(I)*TVB_R_ASTAR
            ENDDO
            
        ENDIF
        
            IF(PIPR_I_T<=1) THEN
                UF_R_YN=0.0
                UF_R_YPP=0.0
                UF_R_YP=0.0
            ENDIF 
            UF_R_TAU=0.0
            DO I=1,TVB_I_NUM
                UF_R_YN(PIPR_I_K,I)=UF_R_YPP(PIPR_I_K,I)*EXP(-TVB_R_N(I)*UF_R_TAU0)+TVB_R_M(I)*(1-EXP(-TVB_R_N(I)*UF_R_TAU0))/(TVB_R_N(I)*UF_R_TAU0)*(PIPR_R_QP(PIPR_I_K)-PIPR_R_QPP(PIPR_I_K))/PIPR_R_AS
                !UF_R_YN(PIPR_I_K,I)=UF_R_YPP(PIPR_I_K,I)*EXP(-TVB_R_N(I)*UF_R_TAU0)+TVB_R_M(I)*EXP(-TVB_R_N(I)*0.5*UF_R_TAU0)*(PIPR_R_QP(PIPR_I_K)-PIPR_R_QPP(PIPR_I_K))/PIPR_R_AS     !�������һ�£�����ļ�����΢��һЩ
                UF_R_TAU=UF_R_TAU+4*1000*PIPR_R_NU/PIPR_R_D*UF_R_YN(PIPR_I_K,I)
                UF_R_YPP(PIPR_I_K,I)=UF_R_YP(PIPR_I_K,I)
           		UF_R_YP(PIPR_I_K,I)=UF_R_YN(PIPR_I_K,I)
            ENDDO
            
            

        
    
END