import { Header } from '@/components/common/header';
import FarmerView from '@/components/farmer-view';
import PolicymakerDashboard from '@/components/policymaker-dashboard';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { loadStationList } from '@/lib/data'; 
import AiAssistant from '@/components/ai-assistant';

export default async function Home() {
 const stations = await loadStationList();

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        <div className="container mx-auto p-4 md:p-8">
          <Tabs defaultValue="policymaker" className="w-full">
            <TabsList className="grid w-full grid-cols-2 md:w-[400px]">
              <TabsTrigger value="policymaker">Policymaker Dashboard</TabsTrigger>
              <TabsTrigger value="farmer">Farmer View</TabsTrigger>
            </TabsList>
            <TabsContent value="policymaker" className="mt-6">
              <PolicymakerDashboard stations={stations} />
            </TabsContent>
            <TabsContent value="farmer" className="mt-6">
              <FarmerView stations={stations} />
            </TabsContent>
          </Tabs>
        </div>
      </main>
      <AiAssistant />
    </div>
  );
}