all:	datasim_utility detector_features sc_kinematics
	@echo "finished building all targets"

datasim_utility:
	@echo "+---------------------------------+"
	@echo "| building datasim_utility target |"
	@echo "+---------------------------------+"
	make -f ./Utilities/Makefile
	sudo make install -f ./Utilities/Makefile
	@echo "+------------------------------+"
	@echo "| built datasim_utility target |"
	@echo "+------------------------------+"
	@echo

detector_features:
	@echo "+-----------------------------------+"
	@echo "| building detector_features target |"
	@echo "+-----------------------------------+"
	make -f ./Detector_features/Makefile
	sudo make install -f ./Detector_features/Makefile
	@echo "+--------------------------------+"
	@echo "| built detector_features target |"
	@echo "+--------------------------------+"
	@echo

sc_kinematics:
	@echo "+--------------------------------+"
	@echo "| building sc_kinematics target |"
	@echo "+--------------------------------+"
	make -f ./Sc_kinematics/Makefile
	sudo make install -f ./Sc_kinematics/Makefile
	@echo "+-----------------------------+"
	@echo "| built sc_kinematics target |"
	@echo "+-----------------------------+"
	@echo

clean:
	make -f ./Utilities/Makefile clean
	make -f ./Detector_features/Makefile clean
	make -f ./Sc_kinematics/Makefile clean

uninstall:
	sudo make -f ./Utilities/Makefile uninstall
	sudo make -f ./Detector_features/Makefile uninstall
	sudo make -f ./Sc_kinematics/Makefile uninstall
