
SUBROUTINE SUB_UF(PIPR_I_NS                            ,&!管段分段数                 整型         (    /    )    //输入参数
                  PIPR_I_T                             ,&!计算时步个数                      整型         (    s    )    //输入参数
                  PIPR_I_J                             ,&!管道节点                          整型         (    /    )    //输入参数
                  PIPR_I_K                             ,&!PIPR_I_J左(右)节点                整型         (    /    )    //输入参数
                  PIPR_R_D                             ,&!管道直径                          实型         (    m    )    //输入参数
                  PIPR_R_A                             ,&!管道波速                          实型         (   m/s   )    //输入参数
                  PIPR_R_AS                            ,&!管道面积                          实型         (   m2    )    //输入参数
                  PIPR_R_DT                            ,&!时间步长                          实型         (    s    )    //输入参数
                  PIPR_R_QP                            ,&!前一时刻流量                      实型         (   m3/s  )    //输入参数
                  PIPR_R_QPP                           ,&!前前时刻流量                      实型         (   m3/s  )    //输入参数
                  PIPR_R_REP                           ,&!上一时刻雷诺数                    实型         (    /    )    //输入参数
                  UF_R_YN                              ,&!卷积积分本时刻Y                   实型         (    /    )    //输入参数/输出参数
                  UF_R_YP                              ,&!卷积积分前时刻Y                   实型         (    /    )    //输入参数/输出参数
                  UF_R_YPP                             ,&!卷积积分前前时刻Y                 实型         (    /    )    //输入参数/输出参数
                  UF_R_TAU)                              !壁面剪切力                        实型         (    /    )    //输出参数

    IMPLICIT NONE
    
    INTEGER:: PIPR_I_NS                               !所有计算节点数量                  整型         (    /    )    //输入参数
    INTEGER::PIPR_I_T                                 !计算时步个数                      整型         (    s    )    //输入参数
    INTEGER::PIPR_I_J                                 !管道节点                          整型         (    /    )    //输入参数
    INTEGER::PIPR_I_K                                 !PIPR_I_J左(右)节点                整型         (    /    )    //输入参数
    REAL(KIND=8)::PIPR_R_D                                    !管道直径                          实型         (    m    )    //输入参数
    REAL(KIND=8)::PIPR_R_A                                    !管道波速                          实型         (   m/s   )    //输入参数
    REAL(KIND=8)::PIPR_R_AS                                   !管道面积                          实型         (   m2    )    //输入参数
    REAL(KIND=8)::PIPR_R_DT                                   !时间步长                          实型         (    s    )    //输入参数
    REAL(KIND=8)::PIPR_R_QP(1:PIPR_I_NS+1)                      !前一时刻流量                      实型         (   m3/s  )    //输入参数
    REAL(KIND=8)::PIPR_R_QPP(1:PIPR_I_NS+1)                     !前前时刻流量                      实型         (   m3/s  )    //输入参数
    REAL(KIND=8)::PIPR_R_REP(1:PIPR_I_NS+1)                     !上一时刻雷诺数                    实型         (    /    )    //输入参数 
                                                                                   
    REAL(KIND=8)::MIAB_R_K                                    !MIAB模型系数                      实型         (    /    )    //中间参数
    REAL(KIND=8)::PIPR_R_NU                                   !运动粘度                          实型         (    /    )    //中间参数
    REAL(KIND=8)::PIPR_R_KS                                   !管道粗糙度                        实型         (    /    )    //中间参数
    REAL(KIND=8)::TVB_R_N(1:17)                               !TVB模型系数                       实型         (    /    )    //中间参数
    REAL(KIND=8)::TVB_R_M(1:17)                               !TVB模型系数                       实型         (    /    )    //中间参数  
    REAL(KIND=8)::UF_R_TAU0                                   !初始壁面剪切力                    实型         (    /    )    //中间参数
    REAL(KIND=8)::TVB_R_ASTAR                                 !TVB参数ASTAR                      实型         (    /    )    //中间参数
    REAL(KIND=8)::TVB_R_BSTAR                                 !TVB参数BSTAR                      实型         (    /    )    //中间参数
    INTEGER::TVB_I_NUM                                        !循环变量                          整型         (    /    )    //中间参数
    INTEGER::I                                                !循环变量                          整型         (    /    )    //中间参数
    
    REAL(KIND=8)::UF_R_YN(1:PIPR_I_NS+1,1:17)                   !卷积积分本时刻Y                   实型         (    /    )    //输入参数/输出参数
    REAL(KIND=8)::UF_R_YP(1:PIPR_I_NS+1,1:17)                   !卷积积分前时刻Y                   实型         (    /    )    //输入参数/输出参数
    REAL(KIND=8)::UF_R_YPP(1:PIPR_I_NS+1,1:17)                  !卷积积分前前时刻Y                 实型         (    /    )    //输入参数/输出参数
    REAL(KIND=8)::UF_R_TAU                                    !壁面剪切力                        实型         (    /    )    //输出参数
     
    
 
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
                !UF_R_YN(PIPR_I_K,I)=UF_R_YPP(PIPR_I_K,I)*EXP(-TVB_R_N(I)*UF_R_TAU0)+TVB_R_M(I)*EXP(-TVB_R_N(I)*0.5*UF_R_TAU0)*(PIPR_R_QP(PIPR_I_K)-PIPR_R_QPP(PIPR_I_K))/PIPR_R_AS     !两个结果一致，上面的计算稍微快一些
                UF_R_TAU=UF_R_TAU+4*1000*PIPR_R_NU/PIPR_R_D*UF_R_YN(PIPR_I_K,I)
                UF_R_YPP(PIPR_I_K,I)=UF_R_YP(PIPR_I_K,I)
           		UF_R_YP(PIPR_I_K,I)=UF_R_YN(PIPR_I_K,I)
            ENDDO
            
            

        
    
END