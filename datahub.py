import paho.mqtt.client as mqtt
import json

def sendMessage(username, password, tagName, value):
    hostname = "rabbitmq-pub.education.wise-paas.com" 
    port = 1883  
    username = "7LeD0ox8MZeY:2VQ66jwX6nQZ" 
    password = "nBoyDSRbbRsyX2S2yQbt"
    publish_topic = "/wisepaas/scada/14b290c2-d759-4247-ae6e-d55e64656aba/data" 
    payload = {"d": {"Device1": {tagName: str(value)}}}

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
