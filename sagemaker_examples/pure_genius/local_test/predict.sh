#!/bin/bash

payload=$1

curl --data-binary @${payload} -H "Content-Type: text/csv" -v http://localhost:8080/invocations

