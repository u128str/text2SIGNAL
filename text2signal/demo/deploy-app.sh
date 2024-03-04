#!/bin/bash

if [[ "$1" == "--login" ]]; then
  cf login -a https://api.cf.eu12.hana.ondemand.com --sso -o processaicanary_demo -s sapphire
else
  cf target -o processaicanary_demo -s sapphire
fi

cf push