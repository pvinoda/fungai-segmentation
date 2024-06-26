# FUNGAI END-TO-END DEMO

## Basic Steps for Setup and Customization

1. Mount NCSU Research Storage on your frontend or testing machine
    * NCSU Research Storage Docs: (https://research.oit.ncsu.edu/docs/storage/)
    * Contact Research Storage Support: (oit_research_storage@help.ncsu.edu)
    > Note: having a Research Storage mount on a linux machine that will function without an active login (like on a server) will require some addition configuration. Contact Research Storage or the Research Facilitation Service for more information.
2. Access the HPC Cluster Hazel as the **fungai** user
    * Logging into Hazel: (https://hpc.ncsu.edu/Documents/Login.php)
    * Logging in as **fungai**: `sudo -u fungai bash`
    * Contact HPC Support: (oit_hpc@help.ncsu.edu)
3. Install and install and configure your python executable (fungai.py serving as a placeholder) to function as an LSF job
    * LSF Jobs: (https://hpc.ncsu.edu/Documents/LSF.php)
    * Anaconda on Hazel: (https://hpc.ncsu.edu/Software/Apps.php?app=Conda)
    * Intro to NCSU HPC (some details are outdated): (https://www.youtube.com/watch?v=Kj8LGsjVBWA&list=PLskHZ4tWojE2_j1WfmALjL3WolXSNHG6a)
    > I recommend installing your conda environments on RS so you have permanent access without running out of storage on your home: **conda create --prefix /path/on/research/storage**. Fungai can read/use anything within `/rs1/researchers/o/oargell/`.
4. Modify process.sh and your config file to contain information that stays consistent between runs. 
    > Try to optimize **process.sh** to pull the minimum resources necessary for the job, it will reduce the potential wait time.
5. Modify check_and_submit.sh to supply information that changes between runs and submit the jobs
6. Once check_and_submit.sh works as intended, ssh onto **servxfer** as **fungai** and start a cronjob that runs check_and_submit.sh every minute
    * `sudo -u fungai bash`
    * `ssh servxfer`
    * `crontab -e`
