#!/bin/bash

mv /gpfs_common/share02/fungai/job_scratch/* /rs1/researchers/d/dmdevrie/fungai_RS/fungai_stage/
rm -rf /rs1/researchers/d/dmdevrie/fungai_RS/fungai_jobs/* 
mv /rs1/researchers/d/dmdevrie/fungai_RS/fungai_out/*/*png /rs1/researchers/d/dmdevrie/fungai_RS/fungai_stage/
rm /rs1/researchers/d/dmdevrie/fungai_RS/fungai_stage/*EDGES.png
rm -rf /rs1/researchers/d/dmdevrie/fungai_RS/fungai_out/*
