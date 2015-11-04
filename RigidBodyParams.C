/*
 *  RigidBodyParams.C
 */
 
#define DEBUGM
#include "RigidBodyParams.h"
#include "MStream.h"

#include "DataStream.h"
#include "InfoStream.h"

#include "Debug.h"
#include <stdio.h>

RigidBodyParams* RigidBodyParamsList::find_key(const char* key)
{
    RBElem* cur = head;
    RBElem* found = NULL;
    RigidBodyParams* result = NULL;
    
    while (found == NULL && cur != NULL) {
       if (!strcasecmp((cur->elem).rigidBodyKey,key)) {
        found = cur;
      } else {
        cur = cur->nxt;
      }
    }
    if (found != NULL) {
      result = &(found->elem);
    }
    return result;
}
  
int RigidBodyParamsList::index_for_key(const char* key)
{
    RBElem* cur = head;
    RBElem* found = NULL;
    int result = -1;
    
    int idx = 0;
    while (found == NULL && cur != NULL) {
       if (!strcasecmp((cur->elem).rigidBodyKey,key)) {
        found = cur;
      } else {
        cur = cur->nxt;
	idx++;
      }
    }
    if (found != NULL) {
	result = idx;
    }
    return result;
}
  
RigidBodyParams* RigidBodyParamsList::add(const char* key) 
{
    // If the key is already in the list, we can't add it
    if (find_key(key)!=NULL) {
      return NULL;
    }
    
    RBElem* new_elem = new RBElem();
    int len = strlen(key);
    RigidBodyParams* elem = &(new_elem->elem);
    elem->rigidBodyKey = new char[len+1];
    strncpy(elem->rigidBodyKey,key,len+1);
    elem->mass = NULL;
    elem->inertia = Vector(NULL);
    elem->langevin = NULL;
    elem->temperature = NULL;
    elem->transDampingCoeff = Vector(NULL);
    elem->rotDampingCoeff = Vector(NULL);
    elem->gridList.clear();
    elem->position = Vector(NULL);
    elem->velocity = Vector(NULL);
    elem->orientation = Tensor();
    elem->orientationalVelocity = Vector(NULL);
    
    elem->next = NULL;
    new_elem->nxt = NULL;
    if (head == NULL) {
      head = new_elem;
    }
    if (tail != NULL) {
      tail->nxt = new_elem;
      tail->elem.next = elem;
    }
    tail = new_elem;
    n_elements++;
    
    return elem;
}  

const void RigidBodyParams::print() {
    iout << iINFO
	 << "printing RigidBodyParams("<<rigidBodyKey<<"):"
	 <<"\n\t" << "mass: " << mass
	 <<"\n\t" << "inertia: " << inertia
	 <<"\n\t" << "langevin: " << langevin
	 <<"\n\t" << "temperature: " << temperature
	 <<"\n\t" << "transDampingCoeff: " << transDampingCoeff
	 <<"\n\t" << "position: " << position
	 <<"\n\t" << "orientation: " << orientation
	 <<"\n\t" << "orientationalVelocity: " << orientationalVelocity
	 << "\n"  << endi;

}
const void RigidBodyParamsList::print() {
    iout << iINFO << "Printing " << n_elements << " RigidBodyParams\n" << endi;
	
    RigidBodyParams *elem = get_first();
    while (elem != NULL) {
	elem->print();
	elem = elem->next;
    }
}
const void RigidBodyParamsList::print(char *s) {
    iout << iINFO << "("<<s<<") Printing " << n_elements << " RigidBodyParams\n" << endi;
	
    RigidBodyParams *elem = get_first();
    while (elem != NULL) {
	elem->print();
	elem = elem->next;
    }
}

void RigidBodyParamsList::pack_data(MOStream *msg) {
    DebugM(4, "Packing rigid body parameter list\n");
    print();

    int i = n_elements;
    msg->put(n_elements);
    
    RigidBodyParams *elem = get_first();
    while (elem != NULL) {
    	DebugM(4, "Packing a new element\n");

    	int len;
	Vector v;
	
	len = strlen(elem->rigidBodyKey) + 1;
    	msg->put(len);
    	msg->put(len,elem->rigidBodyKey);
	msg->put(elem->mass);
	
	// v = elem->
	msg->put(&(elem->inertia));
	msg->put( (elem->langevin?1:0) ); 
	msg->put(elem->temperature);
	msg->put(&(elem->transDampingCoeff));
	msg->put(&(elem->rotDampingCoeff));
    	
	// elem->gridList.clear();
	
	msg->put(&(elem->position));
	msg->put(&(elem->velocity));
	// Tensor data = elem->orientation;
	msg->put( & elem->orientation );
	msg->put(&(elem->orientationalVelocity)) ;
	
	i--;
	elem = elem->next;
    }
    if (i != 0)
      NAMD_die("MGridforceParams message packing error\n");
}
void RigidBodyParamsList::unpack_data(MIStream *msg) {
    DebugM(4, "Could be unpacking rigid body parameterlist (not used & not implemented)\n");

    int elements;
    msg->get(elements);

    for(int i=0; i < elements; i++) {
    	DebugM(4, "Unpacking a new element\n");

	int len;
	msg->get(len);
	char *key = new char[len];
	msg->get(len,key);
	RigidBodyParams *elem = add(key);
	delete [] key;
	
	msg->get(&(elem->inertia));

	int j;
	msg->get(j);
	elem->langevin = (j != 0); 
	
	msg->get(elem->temperature);
	msg->get(&(elem->transDampingCoeff));
	msg->get(&(elem->rotDampingCoeff));
    	
	// elem->gridList.clear();
	
	msg->get(&(elem->position));
	msg->get(&(elem->velocity));
	msg->get( & elem->orientation );
	msg->get(&(elem->orientationalVelocity)) ;
	
	elem = elem->next;
    }

    DebugM(4, "Finished unpacking rigid body parameter list\n");
    print();

}
