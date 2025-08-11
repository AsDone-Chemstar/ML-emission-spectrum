#!/bin/bash

sub=$(($1*4/5))
valid=$(($1/5))
cp itrain.dat itrain.orig.dat
head -n $1 itrain.orig.dat > itrain.dat
head -n $sub itrain.dat > isubtrain.dat
tail -n $valid itrain.dat > ivalidate.dat
