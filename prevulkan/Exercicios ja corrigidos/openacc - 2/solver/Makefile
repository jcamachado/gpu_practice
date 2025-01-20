choose:
	@echo make c++ or make fortran or make both

CPP=g++
FC=pgfortran
TIMER=/usr/bin/time
OPT=
NOPT=-Ofast -Minfo=opt $(OPT)
ARGS=1000

c++: jsolvec.exe
	$(TIMER) ./jsolvec.exe $(ARGS)
jsolvec.exe: jsolvec.cpp
	$(CPP) -o $@ $< $(NOPT)

fortran: jsolvef.exe
	$(TIMER) ./jsolvef.exe $(STEPS)
jsolvef.exe: jsolvef.F90
	$(FC) -o $@ $< $(NOPT)

both: c++ fortran

clean:
	rm -f *.o *.exe *.s *.mod a.out
