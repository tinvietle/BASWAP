import paho.mqtt.client as mqtt
import json


hostname = "rabbitmq-pub.education.wise-paas.com" 
port = 1883  
username = "7LeD0ox8MZeY:ENeoLMO663q6" 
password = "4TJHvPpWyQJ3AnsLoh3u"
publish_topic = "/wisepaas/scada/14b290c2-d759-4247-ae6e-d55e64656aba/data" 
payload = {"d": {"Device1": {"Prompt_SOS": "The salinity level is 1.2. Since it's above 1, the water is saline, so close the gate."}}} 


payload_str = json.dumps(payload)


client = mqtt.Client()


client.username_pw_set(username, password)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}")


client.on_connect = on_connect


client.connect(hostname, port, keepalive=60)


result = client.publish(publish_topic, payload_str)


status = result.rc
if status == 0:
    print(f"Message sent to topic `{publish_topic}`")
else:
    print(f"Failed to send message to topic `{publish_topic}`")

client.disconnect()
