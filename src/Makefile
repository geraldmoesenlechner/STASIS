all:	datasim_utility detector_features dpu_algorithms
	@echo "finished building all targets"

datasim_utility:
	@echo "+---------------------------------+"
	@echo "| building datasim_utility target |"
	@echo "+---------------------------------+"
	make -f ./Makefile-utilities.mk
	sudo make install -f ./Makefile-utilities.mk
	@echo "+------------------------------+"
	@echo "| built datasim_utility target |"
	@echo "+------------------------------+"
	@echo

detector_features:
	@echo "+-----------------------------------+"
	@echo "| building detector_features target |"
	@echo "+-----------------------------------+"
	make -f ./Makefile-dfl.mk
	sudo make install -f ./Makefile-dfl.mk
	@echo "+--------------------------------+"
	@echo "| built detector_features target |"
	@echo "+--------------------------------+"
	@echo

clean:
	make -f ./Makefile-utilities.mk clean
	make -f ./Makefile-dfl.mk clean

uninstall:
	sudo make -f ./Makefile-utilities.mk uninstall
	sudo make -f ./Makefile-dfl.mk uninstall
