/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/
#ifndef MIN_DEBUG_LEVEL
#define MIN_DEBUG_LEVEL 5
#endif
#define DEBUGM
#include "Debug.h"

#include <iostream>
#include <typeinfo>

#include "RigidBody.h"
#include "RigidBodyController.h"
#include "RigidBodyMsgs.h"
// #include "Vector.h"
#include "Node.h"
#include "ReductionMgr.h"
//#include "Broadcasts.h"
#include "ComputeMgr.h"
#include "Random.h"
#include "Output.h"

#include "SimParameters.h"
#include "RigidBodyParams.h"
#include "Molecule.h"
// #include "InfoStream.h"
#include "common.h"
#include <errno.h>


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
