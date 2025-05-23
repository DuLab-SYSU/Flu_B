Code and data needed to reproduce the results in manuscript "Unraveling the mechanism behind the potential extinction of the B/Yamagata lineage of influenza B viruses"  

./code includes:  
--PREDAC folder includes two subfolders BV and BY. BV includes python scripts to construct the antigenic prediction models for B/Victoria while BY includes python scripts to construct the antigenic prediction models for B/Yamagata.  
--dynamic_model folder includes python scripts to construct the two-lineage transmission model for B/Victoria and B/Yamagata:  
./flu_b_mcmc_parallel.py was used to estimiate the parameters by MCMC.  
./flu_b_search_r0.py was used to determine the value of R0.  
./flu_b_ci.py was used to determine the 95% CI of simulated cases data.  
All scripts with the command suffix were used to run the corresponding above scripts.

./data includes the related dataset used in the corresponding scripts in ./code folder or the results.

Note: As the manuscript has not yet been published, the full code and data are not publicly available at this time. We will release all code and related data upon publication.
