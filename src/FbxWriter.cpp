///////////////////////////////////////////////////////////////////////////////
//               Dem Bones - Skinning Decomposition Library                  //
//         Copyright (c) 2019, Electronic Arts. All rights reserved.         //
///////////////////////////////////////////////////////////////////////////////



#include "FbxWriter.h"
#include "FbxShared.h"
#include "LogMsg.h"
#include <sstream>
#include <Eigen/Dense>
#include <DemBones/MatBlocks.h>
#include <set>

using namespace std;
using namespace Eigen;

#define err(msgStr) {msg(1, msgStr); return false;}

class FbxSceneExporter: public FbxSceneShared {
public:
	FbxSceneExporter(bool embedMedia=true): FbxSceneShared(false) {
		(*(lSdkManager->GetIOSettings())).SetBoolProp(EXP_FBX_EMBEDDED, embedMedia);
	}

	//https://help.autodesk.com/view/FBX/2017/ENU/?guid=__cpp_ref__export_scene01_2main_8cxx_example_html
	//https://help.autodesk.com/view/FBX/2017/ENU/?guid=__cpp_ref__switch_binding_2main_8cxx_example_html
	bool save(const string& fileName) {
		// Create an exporter.
		FbxExporter* lExporter=FbxExporter::Create(lSdkManager, "");

		// Get the appropriate file format.
		int lFileFormat=lSdkManager->GetIOPluginRegistry()->GetNativeWriterFormat();

		// Initialize the exporter.
		if (!lExporter->Initialize(fileName.c_str(), lFileFormat, lSdkManager->GetIOSettings())) err("Call to FbxExporter::Initialize() failed.\n");

		// Export the scene to the file.
		lExporter->Export(lScene);

		lExporter->Destroy();

		return true;
	}

	void createMesh(const MatrixXd verts, const vector<vector<int>> faces){
		FbxMesh* pmesh = FbxMesh::Create(lSdkManager,"mesh");
		pmesh->InitControlPoints(verts.cols());	
		FbxVector4* pControlPoints = pmesh->GetControlPoints();
		
		// Set vertices
		for(auto i=0;i<verts.cols();i++){
			FbxVector4 lControlPoint(verts(0,i), verts(1,i), verts(2,i));
			pControlPoints[i] = lControlPoint;
		}

		// Set faces
		for(auto i=0;i<faces.size();i++){
			pmesh->BeginPolygon(-1.-1,false);
			for(auto j=0;j<faces[i].size();j++)
				pmesh->AddPolygon(faces[i][j]);
			pmesh->EndPolygon();
		}	

		// Create node in scene
		FbxNode* pNode = FbxNode::Create(lScene,"mesh");
		pNode->SetNodeAttribute(pmesh);

		// Add node to scene root
		lScene->GetRootNode()->AddChild(pNode);

		return;
	}

	void createJoints(const vector<string>& name, const ArrayXi& parent, double radius) {
		FbxNode* lRoot=NULL;
		for (int j=0; j!=name.size(); j++)
			if (parent(j)==-1) {
				FbxSkeleton* lSkeletonAttribute=FbxSkeleton::Create(lScene, name[j].c_str());
				lSkeletonAttribute->SetSkeletonType(((parent==j).count()>0)?FbxSkeleton::eRoot:FbxSkeleton::eLimb);
				lSkeletonAttribute->Size.Set(radius);
				FbxNode* lSkeleton=FbxNode::Create(lScene, name[j].c_str());
				lSkeleton->SetNodeAttribute(lSkeletonAttribute);
				lSkeleton->SetRotationOrder(FbxNode::eSourcePivot, eEulerXYZ);
				lScene->GetRootNode()->AddChild(lSkeleton);
				lRoot=lSkeleton;
			}

		for (int j=0; j!=name.size(); j++) 
			if (parent(j)!=-1) {
				FbxSkeleton* lSkeletonAttribute=FbxSkeleton::Create(lScene, name[j].c_str());
				lSkeletonAttribute->SetSkeletonType(FbxSkeleton::eLimb);
				lSkeletonAttribute->Size.Set(radius);
				FbxNode* lSkeleton=FbxNode::Create(lScene, name[j].c_str());
				lSkeleton->SetNodeAttribute(lSkeletonAttribute);
				lSkeleton->SetRotationOrder(FbxNode::eSourcePivot, eEulerXYZ);
				lRoot->AddChild(lSkeleton);
			}
	}

	void addToCurve(const VectorXd& val, const VectorXd& fTime, FbxAnimCurve* lCurve) {
		lCurve->KeyModifyBegin(); 
		int idx=0;
		int nFr=(int)fTime.size();
		FbxTime lTime;
		for (int k=0; k<nFr; k++) {
			lTime.SetSecondDouble(fTime(k));
			idx=lCurve->KeyAdd(lTime);
			lCurve->KeySetValue(idx, (float)val(k));
			lCurve->KeySetInterpolation(idx, FbxAnimCurveDef::eInterpolationCubic);
			lCurve->KeySetTangentMode(idx, FbxAnimCurveDef::eTangentAuto);
		}
		lCurve->KeyModifyEnd();
	}

	void setJoints(const vector<string>& name, const VectorXd& fTime, const MatrixXd& lr, const MatrixXd& lt, const MatrixXd& lbr, const MatrixXd& lbt) {
		// Animation stack & layer.
		// msg(1, "Settig Joints\n");


		FbxString lAnimStackName="demBones";
		FbxAnimStack* lAnimStack=FbxAnimStack::Create(lScene, lAnimStackName);
		FbxAnimLayer* lAnimLayer=FbxAnimLayer::Create(lScene, "Base Layer");
		lAnimStack->AddMember(lAnimLayer);
		// msg(1, "Defining variables Joints\n");


		VectorXd val;

		// msg(1, "Name Size:"<<name.size() << "\n");
		// msg(1, "Val lr:" <<lr.col(0) << "\n");

		for (int j=0; j!=name.size(); j++) {
			// msg(1, "Name:" << name[j].c_str() << "\n");
			// msg(1, "Rot Vals:" << lbr(0, j) <<  lbr(1, j) << lbr(2, j) << "\n");
			// msg(1, "Trans Vals:" << lbt(0, j) <<  lbt(1, j) << lbt(2, j) << "\n");

			FbxNode* lSkeleton=lScene->FindNodeByName(FbxString(name[j].c_str()));
			lSkeleton->LclRotation.Set(FbxDouble3(lbr(0, j), lbr(1, j), lbr(2, j)));
			lSkeleton->LclTranslation.Set(FbxDouble3(lbt(0, j), lbt(1, j), lbt(2, j)));
			// msg(1, "Val lr:" <<lr << "\n");

			val=lr.col(j);
			// msg(1, "Val lr:" <<lr.col(j) << "\n");
			
			addToCurve(Map<VectorXd, 0, InnerStride<3>>(val.data(), val.size()/3), fTime, lSkeleton->LclRotation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_X, true));
			addToCurve(Map<VectorXd, 0, InnerStride<3>>(val.data()+1, val.size()/3), fTime, lSkeleton->LclRotation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_Y, true));
			addToCurve(Map<VectorXd, 0, InnerStride<3>>(val.data()+2, val.size()/3), fTime, lSkeleton->LclRotation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_Z, true));
			val=lt.col(j);
			// msg(1, "Val lt:" << lt.col(j) << "\n");

			addToCurve(Map<VectorXd, 0, InnerStride<3>>(val.data(), val.size()/3), fTime, lSkeleton->LclTranslation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_X, true));
			addToCurve(Map<VectorXd, 0, InnerStride<3>>(val.data()+1, val.size()/3), fTime, lSkeleton->LclTranslation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_Y, true));
			addToCurve(Map<VectorXd, 0, InnerStride<3>>(val.data()+2, val.size()/3), fTime, lSkeleton->LclTranslation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_Z, true));
		}

	
		// msg(1, "Settig Joints Completed\n");

	}



	
	void setSkinCluster(const vector<string>& name, const SparseMatrix<double>& w, const MatrixXd& gb) {
		FbxMesh* lMesh=firstMesh(lScene->GetRootNode());

		FbxSkin* lSkin=firstSkin(lMesh);
		if (lSkin==NULL) {
			lSkin=FbxSkin::Create(lScene, "demSkinCluster");
			lMesh->AddDeformer(lSkin);
			lSkin->SetSkinningType(FbxSkin::eLinear);
		}

		//Clear all clusters
		while (lSkin->GetClusterCount()) {
			FbxCluster* lCluster=lSkin->GetCluster(lSkin->GetClusterCount()-1);
			lSkin->RemoveCluster(lCluster);
		}

		//Clear all poses
		while (lScene->GetPoseCount()) lScene->RemovePose(lScene->GetPoseCount()-1);

		//Create a new bind pose
		FbxPose* lPose=FbxPose::Create(lScene, "demBindPose");
		lPose->SetIsBindPose(true);

		FbxAMatrix gMat=lMesh->GetNode()->EvaluateGlobalTransform();
		lPose->Add(lMesh->GetNode(), gMat);

		SparseMatrix<double> wT=w.transpose();
		int nB=(int)name.size();
		set<FbxNode*> added;
		for (int j=0; j<nB; j++) {
			ostringstream s;
			s<<"demCluster"<<j;
			FbxCluster* lCluster=FbxCluster::Create(lScene, s.str().c_str());
			FbxNode* lNode=lScene->FindNodeByName(FbxString(name[j].c_str()));
			
			lCluster->SetLink(lNode);
			lCluster->SetLinkMode(FbxCluster::eTotalOne);
			for (SparseMatrix<double>::InnerIterator it(wT, j); it; ++it) lCluster->AddControlPointIndex((int)it.row(), it.value());
			lCluster->SetTransformMatrix(gMat);

			//Equaivalent to jointMat=lJoint->EvaluateGlobalTransform(), but with better accuracy
			FbxAMatrix jointMat;
			Map<Matrix4d> mm(&jointMat.mData[0][0], 4, 4);
			mm=gb.blk4(0, j);

			lCluster->SetTransformLinkMatrix(jointMat);
			lSkin->AddCluster(lCluster);

			//Add to bind pose
			while ((lNode)&&(added.find(lNode)==added.end())) {
				added.insert(lNode);
				lPose->Add(lNode, lNode->EvaluateGlobalTransform());
				lNode=lNode->GetParent();
			}
		}

		lScene->AddPose(lPose);
	}
};

bool writeFBXs(string fileName,  DemBonesExt<double, float>& model, bool embedMedia) {
	msg(1, "Writing outputs:\n");

	FbxSceneExporter exporter(embedMedia);

	// bool needCreateJoints=(model.boneName.size()==0);
	bool needCreateJoints=true; // Create joint names everytime because saving with new number of bones 
	double radius;

	if (needCreateJoints) {
		model.boneName.resize(model.nB);
		for (int j=0; j<model.nB; j++) {
			ostringstream s;
			s<<"joint"<<j;
			model.boneName[j]=s.str();
		}
		radius=sqrt((model.u-(model.u.rowwise().sum()/model.nV).replicate(1, model.nV)).cwiseAbs().rowwise().maxCoeff().squaredNorm()/model.nS);
	}
	for (int s=0; s<model.nS; s++) {
		msg(1, "Loaded complete mesh:" << s << "\n");

		msg(1, "--> \""<<fileName<<"\" ");
		// Write Mesh
		exporter.createMesh(model.u,model.fv);

		MatrixXd lr, lt, gb, lbr, lbt;

		model.computeRTB(s, lr, lt, gb, lbr, lbt);

		if (needCreateJoints) exporter.createJoints(model.boneName, model.parent, radius);
		msg(1, "Bonename:" << model.boneName.size() << "\n");
		msg(1, "Segment:" << model.fTime.segment(model.fStart(s), model.fStart(s+1)-model.fStart(s)).size() << "\n");
		msg(1, "LR:" << lr.size() << "\n");

	
		exporter.setJoints(model.boneName, model.fTime.segment(model.fStart(s), model.fStart(s+1)-model.fStart(s)), lr, lt, lbr, lbt);

		msg(1, "W:" << model.w.size() << "\n");


		exporter.setSkinCluster(model.boneName, model.w, gb);

		msg(1, "Loaded complete mesh\n");

		if (!exporter.save(fileName)) err("Error on exporting file.\n");
		msg(1, "Saved mesh\n");


		msg(1, "("<<model.fStart(s+1)-model.fStart(s)<<" frames)\n");
	}

	return true;
}
