/* #ifndef MIN_DEBUG_LEVEL */
/* #define MIN_DEBUG_LEVEL 5 */
/* #endif */
/* #define DEBUGM */
/* #include "Debug.h" */

/* #include "RigidBody.h" */
#include "RigidBodyController.h"
#include "Configuration.h"

#include "RigidBodyType.h"
/* #include "Random.h" */

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, String file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), __FILE__, line);
      if (abort) exit(code);
   }
}

/* #include <cuda.h> */
/* #include <cuda_runtime.h> */
/* #include <curand_kernel.h> */


RigidBodyController::RigidBodyController(const Configuration& c) :
conf(c) {

	int numRB = 0;
	// grow list of rbs
	for (int i = 0; i < conf.numRigidTypes; i++) {			
		numRB += conf.rigidBody[i].num;
		std::vector<RigidBody> tmp;
		for (int j = 0; j < conf.rigidBody[i].num; j++) {
			RigidBody r(conf, conf.rigidBody[i]);
			tmp.push_back( r );
		}
		rigidBodyByType.push_back(tmp);
	}

}
RigidBodyController::~RigidBodyController() {
	for (int i = 0; i < rigidBodyByType.size(); i++)
		rigidBodyByType[i].clear();
	rigidBodyByType.clear();
}

void RigidBodyController::updateForces() {
	/*––{ RBTODO }–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––.
	| probably coalesce kernel calls, or move this to a device kernel caller   |
	|                                                                          |
	| - consider removing references (unless they are optimized out!) ---      |
	| - caclulate numthreads && numblocks                                      |
	|                                                                          |
	| all threads in a block should: ----------------------------------------  |
	|   (1) apply the same transformations to get the data point position in a |
	|   destination grid ----------------------------------------------------- |
	|   (2) reduce forces and torques to the same location ------------------- |
	|   (3) ???  ------------------------------------------------------------- |
	|                                                                          |
	| Opportunities for memory bandwidth savings:                              |
	`–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
	// int numBlocks = (num * numReplicas) / NUM_THREADS + (num * numReplicas % NUM_THREADS == 0 ? 0 : 1);
	int numBlocks = 1;
	int numThreads = 32;

	
	// Loop over all pairs of rigid body types
	//   the references here make the code more readable, but they may incur a performance loss
	for (int ti = 0; ti < conf.numRigidTypes; ti++) {
		const RigidBodyType& t1 = conf.rigidBody[ti];
		for (int tj = ti; tj < conf.numRigidTypes; tj++) {
			const RigidBodyType& t2 = conf.rigidBody[tj];
				
			const std::vector<String>& keys1 = t1.densityGridKeys; 
			const std::vector<String>& keys2 = t2.potentialGridKeys;

			// Loop over all pairs of grid keys (e.g. "Elec")
			for(int k1 = 0; k1 < keys1.size(); k1++) {
				for(int k2 = 0; k2 < keys2.size(); k2++) {
					if ( keys1[k1] == keys2[k2] ) {
						// found matching keys => calculate force between all grid pairs
						//   loop over rigid bodies of this type
						const std::vector<RigidBody>& rbs1 = rigidBodyByType[ti];
						const std::vector<RigidBody>& rbs2 = rigidBodyByType[tj];

						for (int i = 0; i < rbs1.size(); i++) {
							for (int j = (ti==tj ? 0 : i); j < rbs2.size(); j++) {
								const RigidBody& rb1 = rbs1[i];
								const RigidBody& rb2 = rbs2[j];
									
								computeGridGridForce<<< numBlocks, numThreads >>>
									(t1.rawDensityGrids[k1], t2.rawPotentialGrids[k2],
									 rb1.getBasis(), rb1.getPosition(),
									 rb2.getBasis(), rb2.getPosition());
							}
						}
					}
				}
			}
		}
	}
		
	// RBTODO: see if there is a better way to sync
	gpuErrchk(cudaDeviceSynchronize());

}

/*
RigidBodyController::RigidBodyController(const NamdState *s, const int reductionTag, SimParameters *sp) : state(s), simParams(sp)
{
    DebugM(2, "Rigid Body Controller initializing" 
    	   << "\n" << endi);

    // initialize each RigidBody
    RigidBodyParams *params =  simParams->rigidBodyList.get_first();
    while (params != NULL) {
    	// check validity of params?
    	RigidBody *rb = new RigidBody(simParams, params);
    	rigidBodyList.push_back( rb );
    	params = params->next;
    }

    // initialize translation and rotation data
    trans.resize( rigidBodyList.size() );
    rot.resize( rigidBodyList.size() );
    for (int i=0; i<rigidBodyList.size(); i++) {
    	trans[i] = rigidBodyList[i]->getPosition();
    	rot[i] = rigidBodyList[i]->getOrientation();
    	// trans.insert( rigidBodyList[i]->getPosition(), i );
    	// rot.insert( rigidBodyList[i]->getOrientation(), i ); 
   }

    random = new Random(simParams->randomSeed);
    // random->split(0,PatchMap::Object()->numPatches()+1);
        
    // inbound communication
    DebugM(3, "RBC::init: requiring reduction "<<reductionTag<<" with "<<6*rigidBodyList.size()<<" elements\n" << endi);
    gridReduction = ReductionMgr::Object()->willRequire(reductionTag ,6*rigidBodyList.size() );

    // outbound communication
    CProxy_ComputeMgr cm(CkpvAccess(BOCclass_group).computeMgr);
    computeMgr = cm.ckLocalBranch();

    if (trans.size() != rot.size())
	NAMD_die("failed sanity check\n");    
    RigidBodyMsg *msg = new RigidBodyMsg;
    msg->trans.copy(trans);	// perhaps .swap() would cause problems
    msg->rot.copy(rot);
    computeMgr->sendRigidBodyUpdate(msg);
}
RigidBodyController::~RigidBodyController() {
    delete gridReduction;
}

void RigidBodyController::print(int step) {
    // modeled after outputExtendedData() in Controller.C
    if ( step >= 0 ) {
	// Write RIGID BODY trajectory file
      if ( (step % simParams->rigidBodyOutputFrequency) == 0 ) {
	  if ( ! trajFile.rdbuf()->is_open() ) {
	      // open file
	      iout << "OPENING RIGID BODY TRAJECTORY FILE\n" << endi;
	      NAMD_backup_file(simParams->rigidBodyTrajectoryFile);
	      trajFile.open(simParams->rigidBodyTrajectoryFile);
	      while (!trajFile) {
		  if ( errno == EINTR ) {
		      CkPrintf("Warning: Interrupted system call opening RIGIDBODY trajectory file, retrying.\n");
		      trajFile.clear();
		      trajFile.open(simParams->rigidBodyTrajectoryFile);
		      continue;
		  }
		  char err_msg[257];
		  sprintf(err_msg, "Error opening RigidBody trajectory file %s",simParams->rigidBodyTrajectoryFile);
		  NAMD_err(err_msg);
	      }
	      trajFile << "# NAMD RigidBody trajectory file" << std::endl;
	      printLegend(trajFile);
	  }
	  printData(step,trajFile);
	  trajFile.flush();    
      }
    
      // Write restart File
      if ( simParams->restartFrequency &&
	   ((step % simParams->restartFrequency) == 0) &&
	   (step != simParams->firstTimestep) )
      {
	  iout << "RIGID BODY: WRITING RESTART FILE AT STEP " << step << "\n" << endi;
	  char fname[140];
	  strcpy(fname, simParams->restartFilename);
	
	  strcat(fname, ".rigid");
	  NAMD_backup_file(fname,".old");
	  std::ofstream restartFile(fname);
	  while (!restartFile) {
	      if ( errno == EINTR ) {
		  CkPrintf("Warning: Interrupted system call opening rigid body restart file, retrying.\n");
		  restartFile.clear();
		  restartFile.open(fname);
		  continue;
	      }
	      char err_msg[257];
	      sprintf(err_msg, "Error opening rigid body restart file %s",fname);
	      NAMD_err(err_msg);
	  }
	  restartFile << "# NAMD rigid body restart file" << std::endl;
	  printLegend(restartFile);
	  printData(step,restartFile);
	  if (!restartFile) {
	      char err_msg[257];
	      sprintf(err_msg, "Error writing rigid body restart file %s",fname);
	      NAMD_err(err_msg);
	  } 
      }
    }
    //  Output final coordinates
    if (step == FILE_OUTPUT || step == END_OF_RUN) {
	int realstep = ( step == FILE_OUTPUT ?
			 simParams->firstTimestep : simParams->N );
	iout << "WRITING RIGID BODY OUTPUT FILE AT STEP " << realstep << "\n" << endi;
	static char fname[140];
	strcpy(fname, simParams->outputFilename);
	strcat(fname, ".rigid");
	NAMD_backup_file(fname);
	std::ofstream outputFile(fname);
	while (!outputFile) {
	    if ( errno == EINTR ) {
		CkPrintf("Warning: Interrupted system call opening rigid body output file, retrying.\n");
		outputFile.clear();
		outputFile.open(fname);
		continue;
	    }
	    char err_msg[257];
	    sprintf(err_msg, "Error opening rigid body output file %s",fname);
	    NAMD_err(err_msg);
	} 
	outputFile << "# NAMD rigid body output file" << std::endl;
	printLegend(outputFile);
	printData(realstep,outputFile);
	if (!outputFile) {
	    char err_msg[257];
	    sprintf(err_msg, "Error writing rigid body output file %s",fname);
	    NAMD_err(err_msg);
	} 
    }

    //  Close trajectory file
    if (step == END_OF_RUN) {
	if ( trajFile.rdbuf()->is_open() ) {
	    trajFile.close();
	    iout << "CLOSING RIGID BODY TRAJECTORY FILE\n" << endi;
	}
    }

}
void RigidBodyController::printLegend(std::ofstream &file) {
        file << "#$LABELS step RigidBodyKey"
		 << " posX  posY  posZ"
		 << " rotXX rotXY rotXZ"
		 << " rotYX rotYY rotYZ"
		 << " rotZX rotZY rotZZ"
		 << " velX  velY  velZ"
		 << " angVelX angVelY angVelZ" << std::endl;
}
void RigidBodyController::printData(int step,std::ofstream &file) {
    iout << "WRITING RIGID BODY COORDINATES AT STEP "<< step << "\n" << endi;
    for (int i; i < rigidBodyList.size(); i++) {
	Vector v =  rigidBodyList[i]->getPosition();
	Tensor t =  rigidBodyList[i]->getOrientation();
	file << step <<" "<< rigidBodyList[i]->getKey()
		 <<" "<< v.x <<" "<< v.y <<" "<< v.z;
	file <<" "<< t.xx <<" "<< t.xy <<" "<< t.xz
		 <<" "<< t.yx <<" "<< t.yy <<" "<< t.yz
		 <<" "<< t.zx <<" "<< t.zy <<" "<< t.zz;
	v = rigidBodyList[i]->getVelocity();
	file <<" "<< v.x <<" "<< v.y <<" "<< v.z;
	v = rigidBodyList[i]->getAngularVelocity();
	file <<" "<< v.x <<" "<< v.y <<" "<< v.z
		 << std::endl;
    }
}

void RigidBodyController::integrate(int step) {
    DebugM(3, "RBC::integrate: step  "<< step << "\n" << endi);
    
    DebugM(1, "RBC::integrate: Waiting for grid reduction\n" << endi);
    gridReduction->require();
  
    const Molecule * mol = Node::Object()->molecule;

    // pass reduction force and torque to each grid
    // DebugM(3, "Summing forces on rigid bodies" << "\n" << endi);
    for (int i=0; i < mol->rbReductionIdToRigidBody.size(); i++) {
	Force f;
	Force t;
	for (int k = 0; k < 3; k++) {
	    f[k] = gridReduction->item(6*i + k);
	    t[k] = gridReduction->item(6*i + k + 3);
	}

	if (step % 100 == 1)
	    DebugM(4, "RBC::integrate: reduction/rb " << i <<":"
		   << "\n\tposition: "
		   << rigidBodyList[mol->rbReductionIdToRigidBody[i]]->getPosition()
		   <<"\n\torientation: "
		   << rigidBodyList[mol->rbReductionIdToRigidBody[i]]->getOrientation()
		   << "\n" << endi);

	DebugM(4, "RBC::integrate: reduction/rb " << i <<": "
	       << "force " << f <<": "<< "torque: " << t << "\n" << endi);
	rigidBodyList[mol->rbReductionIdToRigidBody[i]]->addForce(f);
	rigidBodyList[mol->rbReductionIdToRigidBody[i]]->addTorque(t);
    }
    
    // Langevin 
    for (int i=0; i<rigidBodyList.size(); i++) {
	// continue;  // debug
	if (rigidBodyList[i]->langevin) {
	    DebugM(1, "RBC::integrate: reduction/rb " << i
		   <<": calling langevin" << "\n" << endi);
	    rigidBodyList[i]->addLangevin(
		random->gaussian_vector(), random->gaussian_vector() );
	    // DebugM(4, "RBC::integrate: reduction/rb " << i
	    // 	   << " after langevin: force " << rigidBodyList[i]f <<": "<< "torque: " << t << "\n" << endi);
	    // <<": calling langevin" << "\n" << endi);
	}
    }
    
    if ( step >= 0 && simParams->rigidBodyOutputFrequency &&
	 (step % simParams->rigidBodyOutputFrequency) == 0 ) {
	DebugM(1, "RBC::integrate:integrating for before printing output" << "\n" << endi);
	// PRINT
	if ( step == simParams->firstTimestep ) {
	    print(step);
	    // first step so only start this cycle
	    for (int i=0; i<rigidBodyList.size(); i++)  {
		DebugM(2, "RBC::integrate: reduction/rb " << i
		       <<": starting integration cycle of step "
		       << step << "\n" << endi);
		rigidBodyList[i]->integrate(&trans[i],&rot[i],0);
	    }
	} else {
	    // finish last cycle
	    // DebugM(1, "RBC::integrate: reduction/rb " << i
	    // 	   <<": firststep: calling rb->integrate" << "\n" << endi);
	    for (int i=0; i<rigidBodyList.size(); i++) {
		DebugM(2, "RBC::integrate: reduction/rb " << i
		       <<": finishing integration cycle of step "
		       << step-1 << "\n" << endi);
		rigidBodyList[i]->integrate(&trans[i],&rot[i],1);
	    }
	    print(step);
	    // start this cycle
	    for (int i=0; i<rigidBodyList.size(); i++) {
		DebugM(2, "RBC::integrate: reduction/rb " << i
		       <<": starting integration cycle of step "
		       << step << "\n" << endi);
		rigidBodyList[i]->integrate(&trans[i],&rot[i],0);
	    }
	}
    } else {
	DebugM(1, "RBC::integrate: trans[0] before: " << trans[0] << "\n" << endi);
	if ( step == simParams->firstTimestep ) {
	    // integrate the start of this cycle
	    for (int i=0; i<rigidBodyList.size(); i++) {
		DebugM(2, "RBC::integrate: reduction/rb " << i
		       <<": starting integration cycle of (first) step "
		       << step << "\n" << endi);
		rigidBodyList[i]->integrate(&trans[i],&rot[i],0);
	    }
	} else {
	    // integrate end of last ts and start of this one 
	    for (int i=0; i<rigidBodyList.size(); i++) {
		DebugM(2, "RBC::integrate: reduction/rb " << i
		   <<": ending / starting integration cycle of step "
		   << step-1 << "-" << step << "\n" << endi);
		rigidBodyList[i]->integrate(&trans[i],&rot[i],2);
	    }
	}
	DebugM(1, "RBC::integrate: trans[0] after: " << trans[0] << "\n" << endi);
    }
    
    DebugM(3, "sendRigidBodyUpdate on step: " << step << "\n" << endi);

    if (trans.size() != rot.size())
	NAMD_die("failed sanity check\n");    
    RigidBodyMsg *msg = new RigidBodyMsg;
    msg->trans.copy(trans);	// perhaps .swap() would cause problems
    msg->rot.copy(rot);
    computeMgr->sendRigidBodyUpdate(msg);
}


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
*/
