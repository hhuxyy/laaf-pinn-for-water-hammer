        !COMPILER-GENERATED INTERFACE MODULE: Sun Apr  7 17:27:50 2024
        ! This source file is for reference only and may not completely
        ! represent the generated interface used by the compiler.
        MODULE SUB_UF__genmod
          INTERFACE 
            SUBROUTINE SUB_UF(PIPR_I_NS,PIPR_I_T,PIPR_I_J,PIPR_I_K,     &
     &PIPR_R_D,PIPR_R_A,PIPR_R_AS,PIPR_R_DT,PIPR_R_QP,PIPR_R_QPP,       &
     &PIPR_R_REP,UF_R_YN,UF_R_YP,UF_R_YPP,UF_R_TAU)
              INTEGER(KIND=4) :: PIPR_I_NS
              INTEGER(KIND=4) :: PIPR_I_T
              INTEGER(KIND=4) :: PIPR_I_J
              INTEGER(KIND=4) :: PIPR_I_K
              REAL(KIND=8) :: PIPR_R_D
              REAL(KIND=8) :: PIPR_R_A
              REAL(KIND=8) :: PIPR_R_AS
              REAL(KIND=8) :: PIPR_R_DT
              REAL(KIND=8) :: PIPR_R_QP(1:PIPR_I_NS+1)
              REAL(KIND=8) :: PIPR_R_QPP(1:PIPR_I_NS+1)
              REAL(KIND=8) :: PIPR_R_REP(1:PIPR_I_NS+1)
              REAL(KIND=8) :: UF_R_YN(1:PIPR_I_NS+1,1:17)
              REAL(KIND=8) :: UF_R_YP(1:PIPR_I_NS+1,1:17)
              REAL(KIND=8) :: UF_R_YPP(1:PIPR_I_NS+1,1:17)
              REAL(KIND=8) :: UF_R_TAU
            END SUBROUTINE SUB_UF
          END INTERFACE 
        END MODULE SUB_UF__genmod
