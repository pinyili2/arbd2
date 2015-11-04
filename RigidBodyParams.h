/*
 *  RigidBodyParams.h
 *  
 *
 *  Created by Chris Maffeo on 6/7/2012.
 *  Copyright 2012 _. All rights reserved.
 *
 */
#ifndef RIGIDBODYPARAMS_H
#define RIGIDBODYPARAMS_H

#include "strlib.h"
#include "common.h"
#include "Vector.h"
#include "Tensor.h"
#include "InfoStream.h"
#include "MStream.h"

#include <vector>

class RigidBodyParams {
public:
    RigidBodyParams() {
	rigidBodyKey = 0;
	mass = 0;
	inertia = Vector(0);
	langevin = FALSE;
	temperature = 0;
	transDampingCoeff = Vector(0);
	rotDampingCoeff = Vector(0);
	gridList;
	position = Vector(0);
	velocity = Vector(0);
	orientation = Tensor();
	orientationalVelocity = Vector(0);	   
    }    
    char *rigidBodyKey;
    BigReal mass;
    zVector inertia;
    Bool langevin;
    BigReal temperature;
    zVector transDampingCoeff;
    zVector rotDampingCoeff;
    std::vector<std::string> gridList;
    
    zVector position;
    zVector velocity;
    Tensor orientation;
    zVector orientationalVelocity;

    RigidBodyParams *next;

    const void print();
};


class RigidBodyParamsList {
public:
  RigidBodyParamsList() {
    clear();
  }
  
  ~RigidBodyParamsList() 
  {
    RBElem* cur;
    while (head != NULL) {
      cur = head;
      head = cur->nxt;
      delete cur;
    }
    clear();
  }
  const void print(char *s);
  const void print();

  // The SimParameters bit copy overwrites these values with illegal pointers,
  // So thise throws away the garbage and lets everything be reinitialized
  // from scratch
  void clear() {
    head = tail = NULL;
    n_elements = 0;
  }
  
  RigidBodyParams* find_key(const char* key);  
  int index_for_key(const char* key);
  RigidBodyParams* add(const char* key);
  
  RigidBodyParams *get_first() {
    if (head == NULL) {
      return NULL;
    } else return &(head->elem);
  }
  
  void pack_data(MOStream *msg);  
  void unpack_data(MIStream *msg);
  
  // convert from a string to Bool; returns 1(TRUE) 0(FALSE) or -1(if unknown)
  static int atoBool(const char *s)
  {
    if (!strcasecmp(s, "on")) return 1;
    if (!strcasecmp(s, "off")) return 0;
    if (!strcasecmp(s, "true")) return 1;
    if (!strcasecmp(s, "false")) return 0;
    if (!strcasecmp(s, "yes")) return 1;
    if (!strcasecmp(s, "no")) return 0;
    if (!strcasecmp(s, "1")) return 1;
    if (!strcasecmp(s, "0")) return 0;
    return -1;
  }


private:
  class RBElem {
  public:
    RigidBodyParams elem;
    RBElem* nxt;
  };
  RBElem* head;
  RBElem* tail;
  int n_elements;

};

#endif
