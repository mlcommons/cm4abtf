#export CM_GH_TOKEN=""

docker build --network=host  --build-arg CM_GH_TOKEN="${CM_GH_TOKEN}"   -f "abtf-demo-cpu.Dockerfile"   -t abtf-demo-cpu .
