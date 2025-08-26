#!/usr/bin/env python3
import rclpy, os, json
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Header

try:
    from bbox_msgs.msg import BoundingBoxes as BBs
except ImportError:
    from my_msgs.msg import BoundingBoxes as BBs

from my_msgs.msg import Track, TrackArray

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1)
    ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih
    if inter<=0: return 0.0
    area_a=max(0,ax2-ax1)*max(0,ay2-ay1)
    area_b=max(0,bx2-bx1)*max(0,by2-by1)
    return inter/max(1.0,(area_a+area_b-inter))

class MOT(Node):
    def __init__(self):
        super().__init__('mot_node')
        self.iou_th = self.declare_parameter('iou_th', 0.3).value
        self.max_missed = int(self.declare_parameter('max_missed', 30).value)
        self.save_json = self.declare_parameter('save_dets_json', False).value
        self.json_path = self.declare_parameter('dets_json_path', '/workspace/dets.ndjson').value

        self.sub = self.create_subscription(BBs, '/yolo/bounding_boxes', self.cb, qos_profile_sensor_data)
        qos_rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.pub = self.create_publisher(TrackArray, '/tracks', qos_rel)

        self.next_id=1
        self.tracks={}  # id -> {bbox, class_id, score, age, missed}
        self.f=None
        if self.save_json:
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
            self.f=open(self.json_path,'a',buffering=1)
        self.get_logger().info(f'MOT ready (iou_th={self.iou_th}, max_missed={self.max_missed}, save_json={self.save_json})')

    def cb(self,msg:BBs):
        dets=[]
        for b in msg.boxes:
            dets.append({'bbox':[int(b.xmin),int(b.ymin),int(b.xmax),int(b.ymax)],
                         'score': float(getattr(b,'probability',0.0)),
                         'cls': getattr(b,'class_id','')})
        if self.f is not None:
            self.f.write(json.dumps({
                'sec': getattr(msg.header.stamp,'sec',0),
                'nsec': getattr(msg.header.stamp,'nanosec',0),
                'frame_id': msg.header.frame_id,
                'detections': dets
            })+'\n')

        unmatched=set(self.tracks.keys())
        used=set()
        pairs=[]
        for tid,tr in list(self.tracks.items()):
            best=(-1,-1.0)
            for i,d in enumerate(dets):
                if i in used: continue
                v=iou(tr['bbox'], d['bbox'])
                if v>best[1]: best=(i,v)
            if best[1] >= self.iou_th:
                i_det=best[0]; used.add(i_det); unmatched.discard(tid)
                d=dets[i_det]; tr.update(bbox=d['bbox'], score=d['score'], class_id=d['cls'], age=tr['age']+1, missed=0)

        for i,d in enumerate(dets):
            if i in used: continue
            tid=self.next_id; self.next_id+=1
            self.tracks[tid]=dict(bbox=d['bbox'], class_id=d['cls'], score=d['score'], age=1, missed=0)

        for tid in list(unmatched):
            self.tracks[tid]['missed']+=1
            if self.tracks[tid]['missed']>self.max_missed:
                del self.tracks[tid]

        out=TrackArray(); out.header=msg.header if msg.header else Header()
        for tid,tr in self.tracks.items():
            t=Track(); t.id=int(tid); t.class_id=tr['class_id']; t.score=float(tr['score'])
            x1,y1,x2,y2=tr['bbox']; t.xmin,t.ymin,t.xmax,t.ymax=int(x1),int(y1),int(x2),int(y2)
            out.tracks.append(t)
        self.pub.publish(out)

    def destroy_node(self):
        try:
            if self.f: self.f.close()
        except Exception: pass
        super().destroy_node()

def main():
    rclpy.init(); rclpy.spin(MOT()); rclpy.shutdown()
if __name__=='__main__': main()
